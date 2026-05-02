"""Local embedding-backed tool prefiltering for large native tool catalogs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write, atomic_replace

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("cache") / "tool_retrieval"
DEFAULT_MODEL_CACHE_DIR = DEFAULT_CACHE_DIR / "models"
DEFAULT_INDEX_BACKEND = "faiss"
DEFAULT_INDEX_TYPE = "flat_ip"
DEFAULT_RETRIEVAL_PLATFORMS = ("acp", "cli")


class ToolRetrievalError(RuntimeError):
    """Raised when tool retrieval cannot produce a selected subset."""


@dataclass
class ToolRetrievalResult:
    selected_tools: List[dict]
    selected_names: List[str]
    scores: dict[str, float]
    fallback_reason: Optional[str] = None


@dataclass(frozen=True)
class ToolRetrievalPaths:
    index_path: Path
    metadata_path: Path


@dataclass
class ToolRetrievalIndex:
    index: Any
    metadata: dict[str, Any]
    entries: List[dict[str, Any]]
    vector_dimensions: int
    index_path: Path
    metadata_path: Path


Embedder = Callable[[List[str], dict], Any]

_MODEL_CACHE: dict[tuple[str, str, str, str, bool], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def tool_name(tool: dict) -> str:
    return str((tool or {}).get("function", {}).get("name") or "")


def _schema_property_lines(
    properties: dict[str, Any],
    *,
    prefix: str = "",
    required: Iterable[str] = (),
) -> list[str]:
    lines: list[str] = []
    required_set = set(required or ())
    for prop_name in sorted(properties):
        prop = properties.get(prop_name)
        if not isinstance(prop, dict):
            continue
        path = f"{prefix}.{prop_name}" if prefix else prop_name
        type_text = prop.get("type") or prop.get("format") or ""
        if isinstance(type_text, list):
            type_text = "|".join(str(v) for v in type_text)
        desc = str(prop.get("description") or "").strip()
        marker = " required" if prop_name in required_set else ""
        lines.append(f"parameter {path}{marker}: {type_text} {desc}".strip())

        nested = prop.get("properties")
        if isinstance(nested, dict):
            lines.extend(
                _schema_property_lines(
                    nested,
                    prefix=path,
                    required=prop.get("required") or (),
                )
            )
        items = prop.get("items")
        if isinstance(items, dict) and isinstance(items.get("properties"), dict):
            lines.extend(
                _schema_property_lines(
                    items["properties"],
                    prefix=f"{path}[]",
                    required=items.get("required") or (),
                )
            )
    return lines


def tool_schema_text(tool: dict) -> str:
    """Build stable embedding text from a native function tool schema."""
    function = (tool or {}).get("function") or {}
    name = str(function.get("name") or "")
    parts = [
        f"name: {name}",
        f"description: {str(function.get('description') or '').strip()}",
    ]
    params = function.get("parameters") or {}
    if isinstance(params, dict):
        required = params.get("required") or ()
        properties = params.get("properties") or {}
        if isinstance(properties, dict):
            parts.extend(_schema_property_lines(properties, required=required))
    return "\n".join(part for part in parts if part.strip())


def tool_schema_hash(tools: list[dict]) -> str:
    """Fingerprint complete native tool schemas for cache invalidation."""
    payload = [
        {"name": tool_name(tool), "function": (tool or {}).get("function") or {}}
        for tool in sorted(tools or [], key=tool_name)
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _is_truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", "never"}:
            return False
        return default
    return bool(value)


def tool_retrieval_enabled(config: dict, platform: str | None) -> bool:
    cfg = (config or {}).get("tool_retrieval") or {}
    if not isinstance(cfg, dict):
        return False

    if not _is_truthy(cfg.get("enabled", True), default=True):
        return False

    platforms = cfg.get("platforms", list(DEFAULT_RETRIEVAL_PLATFORMS))
    if platforms in (None, ""):
        return True
    if isinstance(platforms, str):
        platforms = [p.strip() for p in platforms.split(",")]
    if not isinstance(platforms, list):
        return False
    normalized = {str(p).strip().lower() for p in platforms if str(p).strip()}
    return "*" in normalized or (platform or "").strip().lower() in normalized


def _model_name(config: dict) -> str:
    return str((config or {}).get("model") or DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def _device(config: dict) -> str:
    return str((config or {}).get("device") or "cpu").strip() or "cpu"


def _revision(config: dict) -> str:
    return str((config or {}).get("revision") or "").strip()


def _normalize_embeddings(config: dict) -> bool:
    return _is_truthy((config or {}).get("normalize_embeddings", True), default=True)


def _top_k(config: dict) -> int:
    try:
        top_k = int((config or {}).get("top_k", 5))
    except (TypeError, ValueError):
        raise ToolRetrievalError("top_k must be an integer") from None
    if top_k <= 0:
        raise ToolRetrievalError("top_k must be > 0")
    return top_k


def _validate_index_config(config: dict) -> None:
    cfg = config or {}
    _top_k(cfg)
    if str(cfg.get("index_backend") or DEFAULT_INDEX_BACKEND) != DEFAULT_INDEX_BACKEND:
        raise ToolRetrievalError("only the faiss tool retrieval index backend is supported")
    if str(cfg.get("index_type") or DEFAULT_INDEX_TYPE) != DEFAULT_INDEX_TYPE:
        raise ToolRetrievalError("only the flat_ip FAISS tool retrieval index type is supported")


def _cache_dir(config: dict) -> Path:
    raw_dir = (config or {}).get("cache_dir") or str(DEFAULT_CACHE_DIR)
    cache_dir = Path(str(raw_dir)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = get_hermes_home() / cache_dir
    return cache_dir


def _model_cache_dir(config: dict) -> Path:
    raw_dir = (config or {}).get("model_cache_dir") or str(DEFAULT_MODEL_CACHE_DIR)
    cache_dir = Path(str(raw_dir)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = get_hermes_home() / cache_dir
    return cache_dir


def _slug(value: str, default: str = "default") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-_")
    return text or default


def index_artifact_paths(config: dict, schema_hash: str, platform: str | None = None) -> ToolRetrievalPaths:
    """Return profile-scoped FAISS and metadata paths for a schema fingerprint."""
    model_slug = _slug(_model_name(config), "model")
    platform_slug = _slug(platform or "default", "default")
    stem = _slug(schema_hash, "index")
    directory = _cache_dir(config) / platform_slug / model_slug
    return ToolRetrievalPaths(
        index_path=directory / f"{stem}.faiss",
        metadata_path=directory / f"{stem}.meta.json",
    )


def index_file_path(config: dict, platform: str | None = None, schema_hash: str | None = None) -> Path:
    """Backward-compatible helper returning the FAISS index artifact path."""
    return index_artifact_paths(config, schema_hash or "index", platform).index_path


def _normalize_vector(vector: Iterable[Any]) -> list[float]:
    return [float(v) for v in vector]


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if not av or not bv or len(av) != len(bv):
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    norm_a = math.sqrt(sum(x * x for x in av))
    norm_b = math.sqrt(sum(y * y for y in bv))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on install env
        raise ToolRetrievalError(
            "faiss package unavailable; install faiss-cpu to use local tool retrieval"
        ) from exc
    return faiss


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on install env
        raise ToolRetrievalError(
            "numpy package unavailable; install numpy to use local tool retrieval"
        ) from exc
    return np


def _get_sentence_transformer_model(config: dict):
    model_name = _model_name(config)
    device = _device(config)
    revision = _revision(config)
    cache_folder = str(_model_cache_dir(config))
    local_files_only = _is_truthy((config or {}).get("local_files_only", False), default=False)
    key = (model_name, device, cache_folder, revision, local_files_only)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - depends on install env
            raise ToolRetrievalError(
                "sentence-transformers package unavailable; install sentence-transformers "
                "to use local tool retrieval"
            ) from exc

        try:
            kwargs: dict[str, Any] = {
                "device": device,
                "cache_folder": cache_folder,
                "local_files_only": local_files_only,
            }
            if revision:
                kwargs["revision"] = revision
            model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:
            raise ToolRetrievalError(
                f"failed to load local embedding model '{model_name}' on device '{device}': {exc}"
            ) from exc

        _MODEL_CACHE[key] = model
        return model


def _python_matrix(vectors: Any) -> list[list[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    if isinstance(vectors, tuple):
        vectors = list(vectors)
    if not isinstance(vectors, list):
        raise ToolRetrievalError("embedding output was not a matrix")
    if vectors and all(not isinstance(item, (list, tuple)) for item in vectors):
        vectors = [vectors]

    matrix: list[list[float]] = []
    for row in vectors:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise ToolRetrievalError("embedding row was not a vector")
        try:
            values = [float(item) for item in row]
        except (TypeError, ValueError) as exc:
            raise ToolRetrievalError(f"embedding row contained non-numeric values: {exc}") from exc
        if not values:
            raise ToolRetrievalError("empty embedding vector")
        if not all(math.isfinite(value) for value in values):
            raise ToolRetrievalError("embedding vector contained non-finite values")
        matrix.append(values)
    return matrix


def _coerce_embedding_matrix(
    vectors: Any,
    *,
    expected_count: int,
    normalize: bool,
    require_numpy: bool = False,
) -> Any:
    try:
        np = _import_numpy()
    except ToolRetrievalError:
        if require_numpy:
            raise
        matrix = _python_matrix(vectors)
        if len(matrix) != expected_count:
            raise ToolRetrievalError(
                f"embedding count did not match input count: got {len(matrix)}, expected {expected_count}"
            )
        width = len(matrix[0]) if matrix else 0
        if width <= 0 or any(len(row) != width for row in matrix):
            raise ToolRetrievalError("embedding dimensions were empty or inconsistent")
        if normalize:
            normalized = []
            for row in matrix:
                norm = math.sqrt(sum(value * value for value in row))
                if norm == 0.0:
                    raise ToolRetrievalError("embedding vector had zero norm")
                normalized.append([value / norm for value in row])
            matrix = normalized
        return matrix

    try:
        array = np.asarray(vectors, dtype=np.float32)
    except Exception as exc:
        raise ToolRetrievalError(f"embedding output could not be converted to float32: {exc}") from exc
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ToolRetrievalError(f"embedding output must be 2D, got shape {getattr(array, 'shape', None)}")
    if array.shape[0] != expected_count:
        raise ToolRetrievalError(
            f"embedding count did not match input count: got {array.shape[0]}, expected {expected_count}"
        )
    if array.shape[1] <= 0:
        raise ToolRetrievalError("empty embedding vector")
    if not np.isfinite(array).all():
        raise ToolRetrievalError("embedding vector contained non-finite values")
    if normalize:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0.0).any():
            raise ToolRetrievalError("embedding vector had zero norm")
        array = array / norms
    return np.ascontiguousarray(array, dtype=np.float32)


def _matrix_width(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], list):
        return len(matrix[0])
    return 0


def _matrix_rows(matrix: Any) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[0])
    if isinstance(matrix, list):
        return len(matrix)
    return 0


def embed_texts_local(texts: List[str], config: dict) -> Any:
    """Embed text with a cached local SentenceTransformer model."""
    if not texts:
        return []
    model = _get_sentence_transformer_model(config or {})
    try:
        batch_size = int((config or {}).get("batch_size", 32) or 32)
    except (TypeError, ValueError):
        batch_size = 32
    normalize = _normalize_embeddings(config or {})

    try:
        vectors = model.encode(
            texts,
            batch_size=max(1, batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    except Exception as exc:
        raise ToolRetrievalError(f"local embedding model failed to encode text: {exc}") from exc
    return _coerce_embedding_matrix(
        vectors,
        expected_count=len(texts),
        normalize=normalize,
        require_numpy=True,
    )


def preload_embedding_model(config: dict) -> None:
    """Load the configured local embedding model into the process cache."""
    _import_numpy()
    _get_sentence_transformer_model(config or {})


def _tool_entries(tools: list[dict]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for tool in tools or []:
        name = tool_name(tool)
        if not name:
            continue
        text = tool_schema_text(tool)
        entries.append(
            {
                "name": name,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return entries


def _index_metadata(tools: list[dict], config: dict, platform: str | None) -> dict[str, Any]:
    cfg = config or {}
    entries = _tool_entries(tools)
    model = _model_name(cfg)
    return {
        "schema_hash": tool_schema_hash(tools),
        "model": model,
        "embedding_model": model,
        "model_revision": _revision(cfg),
        "device": _device(cfg),
        "backend": "sentence-transformers",
        "index_backend": str(cfg.get("index_backend") or DEFAULT_INDEX_BACKEND),
        "index_type": str(cfg.get("index_type") or DEFAULT_INDEX_TYPE),
        "normalize_embeddings": _normalize_embeddings(cfg),
        "platform": platform or "",
        "tool_count": len(entries),
        "tool_names": [entry["name"] for entry in entries],
        "tool_text_hashes": [entry["text_hash"] for entry in entries],
        "entries": entries,
    }


def _load_metadata(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise ToolRetrievalError(f"tool retrieval metadata is corrupted at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ToolRetrievalError(f"tool retrieval metadata is not an object at {path}")
    return data


def _metadata_matches(metadata: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if metadata.get(key) != value:
            return False
    return True


def _load_index(paths: ToolRetrievalPaths, expected_metadata: dict[str, Any]) -> Optional[ToolRetrievalIndex]:
    metadata = _load_metadata(paths.metadata_path)
    if metadata is None:
        return None
    if not paths.index_path.exists():
        return None
    if not _metadata_matches(metadata, expected_metadata):
        return None

    entries = metadata.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ToolRetrievalError(f"tool retrieval metadata has no entries at {paths.metadata_path}")
    vector_dimensions = metadata.get("vector_dimensions")
    if not isinstance(vector_dimensions, int) or vector_dimensions <= 0:
        raise ToolRetrievalError(f"tool retrieval metadata has invalid vector_dimensions at {paths.metadata_path}")

    faiss = _import_faiss()
    try:
        index = faiss.read_index(str(paths.index_path))
    except Exception as exc:
        raise ToolRetrievalError(f"tool retrieval FAISS index is corrupted at {paths.index_path}: {exc}") from exc

    index_dim = int(getattr(index, "d", 0) or 0)
    index_total = int(getattr(index, "ntotal", 0) or 0)
    if index_dim != vector_dimensions:
        raise ToolRetrievalError(
            f"tool retrieval FAISS dimension mismatch: index has {index_dim}, metadata has {vector_dimensions}"
        )
    if index_total != len(entries):
        raise ToolRetrievalError(
            f"tool retrieval FAISS entry count mismatch: index has {index_total}, metadata has {len(entries)}"
        )
    return ToolRetrievalIndex(
        index=index,
        metadata=metadata,
        entries=entries,
        vector_dimensions=vector_dimensions,
        index_path=paths.index_path,
        metadata_path=paths.metadata_path,
    )


def _write_faiss_index_atomic(faiss: Any, index: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        faiss.write_index(index, str(tmp_path))
        atomic_replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _build_index(
    tools: list[dict],
    config: dict,
    platform: str | None,
    *,
    embedder: Embedder,
    expected_metadata: dict[str, Any],
    paths: ToolRetrievalPaths,
) -> ToolRetrievalIndex:
    cfg = config or {}
    _validate_index_config(cfg)

    tool_texts: list[str] = []
    entries: list[dict[str, str]] = []
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        text = tool_schema_text(tool)
        tool_texts.append(text)
        entries.append(
            {
                "name": name,
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )

    if not entries:
        raise ToolRetrievalError("no tool schemas available for local retrieval")

    vectors = embedder(tool_texts, cfg)
    matrix = _coerce_embedding_matrix(
        vectors,
        expected_count=len(tool_texts),
        normalize=_normalize_embeddings(cfg),
    )
    vector_dimensions = _matrix_width(matrix)
    if vector_dimensions <= 0:
        raise ToolRetrievalError("empty embedding vector")

    faiss = _import_faiss()
    try:
        index = faiss.IndexFlatIP(vector_dimensions)
        index.add(matrix)
    except Exception as exc:
        raise ToolRetrievalError(f"failed to build FAISS tool retrieval index: {exc}") from exc
    if int(getattr(index, "ntotal", 0) or 0) != len(entries):
        raise ToolRetrievalError("FAISS index did not store all tool embeddings")

    metadata = dict(expected_metadata)
    metadata["vector_dimensions"] = vector_dimensions
    metadata["entries"] = entries

    _write_faiss_index_atomic(faiss, index, paths.index_path)
    paths.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(paths.metadata_path, metadata, indent=None, separators=(",", ":"))
    return ToolRetrievalIndex(
        index=index,
        metadata=metadata,
        entries=entries,
        vector_dimensions=vector_dimensions,
        index_path=paths.index_path,
        metadata_path=paths.metadata_path,
    )


def load_or_build_index(
    tools: list[dict],
    config: dict,
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
) -> ToolRetrievalIndex:
    if not tools:
        raise ToolRetrievalError("no tools available")
    cfg = config or {}
    _validate_index_config(cfg)
    if embedder is None:
        # On macOS, importing faiss before torch can initialize a second OpenMP
        # runtime and crash later during SentenceTransformer encoding. Preload
        # the local model first so torch owns the process OpenMP runtime before
        # cached FAISS indexes are read or new indexes are built.
        preload_embedding_model(cfg)
        embedder = embed_texts_local
    expected_metadata = _index_metadata(tools, cfg, platform)
    paths = index_artifact_paths(cfg, expected_metadata["schema_hash"], platform)

    cache_error: ToolRetrievalError | None = None
    try:
        cached = _load_index(paths, expected_metadata)
    except ToolRetrievalError as exc:
        cache_error = exc
        logger.warning("Tool retrieval cache invalid; rebuilding: %s", exc)
    else:
        if cached is not None:
            return cached

    try:
        return _build_index(
            tools,
            cfg,
            platform,
            embedder=embedder,
            expected_metadata=expected_metadata,
            paths=paths,
        )
    except ToolRetrievalError as exc:
        if cache_error is not None:
            raise ToolRetrievalError(
                f"failed to rebuild tool retrieval index after cache failure ({cache_error}): {exc}"
            ) from exc
        raise


def select_tools_for_query(
    tools: list[dict],
    query: str,
    config: dict,
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
    prepared_index: ToolRetrievalIndex | None = None,
) -> ToolRetrievalResult:
    """Return the top-K native tool schemas for a query using a prepared FAISS index."""
    if not tools:
        raise ToolRetrievalError("no tools available")
    query = (query or "").strip()
    if not query:
        raise ToolRetrievalError("tool retrieval query is empty")

    cfg = config or {}
    _validate_index_config(cfg)
    top_k = _top_k(cfg)

    embedder = embedder or embed_texts_local
    retrieval_index = prepared_index or load_or_build_index(
        tools,
        cfg,
        platform,
        embedder=embedder,
    )

    query_vectors = embedder([query], cfg)
    query_matrix = _coerce_embedding_matrix(
        query_vectors,
        expected_count=1,
        normalize=_normalize_embeddings(cfg),
    )
    if _matrix_width(query_matrix) != retrieval_index.vector_dimensions:
        raise ToolRetrievalError(
            f"query embedding dimension mismatch: got {_matrix_width(query_matrix)}, "
            f"expected {retrieval_index.vector_dimensions}"
        )

    k = min(top_k, len(retrieval_index.entries))
    try:
        distances, indices = retrieval_index.index.search(query_matrix, k)
    except Exception as exc:
        raise ToolRetrievalError(f"FAISS tool retrieval search failed: {exc}") from exc

    if hasattr(distances, "tolist"):
        distances = distances.tolist()
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    score_row = distances[0] if distances else []
    index_row = indices[0] if indices else []

    selected_names: list[str] = []
    scores: dict[str, float] = {}
    for raw_score, raw_idx in zip(score_row, index_row):
        idx = int(raw_idx)
        if idx < 0 or idx >= len(retrieval_index.entries):
            continue
        name = str(retrieval_index.entries[idx].get("name") or "")
        if not name:
            continue
        selected_names.append(name)
        scores[name] = float(raw_score)

    if not selected_names:
        raise ToolRetrievalError("no indexed tool scores available")

    by_name = {tool_name(tool): tool for tool in tools}
    selected_tools = [by_name[name] for name in selected_names if name in by_name]
    if not selected_tools:
        raise ToolRetrievalError("retrieval returned no usable tool schemas")
    return ToolRetrievalResult(
        selected_tools=selected_tools,
        selected_names=[tool_name(tool) for tool in selected_tools],
        scores=scores,
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _snippet(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def build_tool_retrieval_query(
    user_message: Any,
    messages: list[dict] | None = None,
    *,
    max_recent_messages: int = 4,
    max_chars: int = 4000,
) -> str:
    """Build a compact, auditable query from the current turn and recent context."""
    parts = ["Current user request:", _snippet(_content_text(user_message), 1600)]
    recent = list(messages or [])[-max_recent_messages:]
    for msg in recent:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue
        if role == "assistant" and msg.get("tool_calls"):
            names = []
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    name = (tc.get("function") or {}).get("name")
                    if name:
                        names.append(str(name))
            if names:
                parts.append("Recent assistant tool calls: " + ", ".join(names))
            continue
        text = _content_text(msg.get("content"))
        if text:
            parts.append(f"Recent {role or 'message'}: {_snippet(text, 800)}")

    query = "\n".join(part for part in parts if part)
    if len(query) > max_chars:
        query = query[-max_chars:]
    return query
