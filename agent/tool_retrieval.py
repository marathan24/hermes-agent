"""Embedding-backed tool prefiltering for large native tool catalogs."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_DIR = Path("cache") / "tool_retrieval"
DEFAULT_INDEX_FILENAME = "index.json"


class ToolRetrievalError(RuntimeError):
    """Raised when tool retrieval cannot produce a selected subset."""


@dataclass
class ToolRetrievalResult:
    selected_tools: List[dict]
    selected_names: List[str]
    scores: dict[str, float]
    fallback_reason: Optional[str] = None


Embedder = Callable[[List[str], dict], List[List[float]]]


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
    """Build stable embedding text from a native OpenAI-format tool schema."""
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


def tool_retrieval_enabled(config: dict, platform: str | None) -> bool:
    cfg = (config or {}).get("tool_retrieval") or {}
    if not isinstance(cfg, dict):
        return False

    enabled = cfg.get("enabled", True)
    if isinstance(enabled, str):
        if enabled.strip().lower() in {"0", "false", "no", "off", "never"}:
            return False
    elif not bool(enabled):
        return False

    platforms = cfg.get("platforms", ["acp"])
    if platforms in (None, ""):
        return True
    if isinstance(platforms, str):
        platforms = [p.strip() for p in platforms.split(",")]
    if not isinstance(platforms, list):
        return False
    normalized = {str(p).strip().lower() for p in platforms if str(p).strip()}
    return "*" in normalized or (platform or "").strip().lower() in normalized


def index_file_path(config: dict) -> Path:
    cfg = config or {}
    raw_dir = cfg.get("cache_dir") or str(DEFAULT_CACHE_DIR)
    cache_dir = Path(str(raw_dir)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = get_hermes_home() / cache_dir
    return cache_dir / str(cfg.get("index_filename") or DEFAULT_INDEX_FILENAME)


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


def embed_texts_openai_compatible(texts: List[str], config: dict) -> List[List[float]]:
    """Embed text through an OpenAI-compatible embeddings endpoint."""
    if not texts:
        return []
    cfg = config or {}
    model = str(cfg.get("model") or DEFAULT_EMBEDDING_MODEL)
    api_key_env = str(cfg.get("api_key_env") or "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    if not api_key and not base_url:
        raise ToolRetrievalError(f"{api_key_env} is not set")

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - import failure depends on env
        raise ToolRetrievalError(f"openai package unavailable: {exc}") from exc

    kwargs: dict[str, Any] = {"api_key": api_key or "dummy-key"}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.embeddings.create(model=model, input=texts)
    return [_normalize_vector(item.embedding) for item in response.data]


def _index_metadata(tools: list[dict], config: dict, platform: str | None) -> dict[str, Any]:
    cfg = config or {}
    return {
        "schema_hash": tool_schema_hash(tools),
        "embedding_model": str(cfg.get("model") or DEFAULT_EMBEDDING_MODEL),
        "platform": platform or "",
        "tool_count": len(tools or []),
        "tool_names": sorted(tool_name(tool) for tool in tools if tool_name(tool)),
    }


def _load_index(path: Path, expected_metadata: dict[str, Any]) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Tool retrieval index read failed: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    entries = data.get("entries")
    if not isinstance(metadata, dict) or not isinstance(entries, list):
        return None
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            return None
    return data


def _build_index(
    tools: list[dict],
    config: dict,
    platform: str | None,
    *,
    embedder: Embedder,
) -> dict:
    texts = [tool_schema_text(tool) for tool in tools]
    vectors = embedder(texts, config)
    if len(vectors) != len(tools):
        raise ToolRetrievalError("embedding count did not match tool count")

    entries = []
    for tool, text, vector in zip(tools, texts, vectors):
        name = tool_name(tool)
        if not name:
            continue
        normalized = _normalize_vector(vector)
        if not normalized:
            raise ToolRetrievalError(f"empty embedding for tool {name}")
        entries.append({"name": name, "text": text, "embedding": normalized})

    if not entries:
        raise ToolRetrievalError("no tool embeddings generated")

    metadata = _index_metadata(tools, config, platform)
    metadata["vector_dimensions"] = len(entries[0]["embedding"])
    data = {"metadata": metadata, "entries": entries}
    path = index_file_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, indent=None, separators=(",", ":"))
    return data


def load_or_build_index(
    tools: list[dict],
    config: dict,
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
) -> dict:
    if not tools:
        raise ToolRetrievalError("no tools available")
    embedder = embedder or embed_texts_openai_compatible
    expected_metadata = _index_metadata(tools, config, platform)
    path = index_file_path(config)
    cached = _load_index(path, expected_metadata)
    if cached is not None:
        return cached
    return _build_index(tools, config, platform, embedder=embedder)


def select_tools_for_query(
    tools: list[dict],
    query: str,
    config: dict,
    platform: str | None = None,
    *,
    embedder: Embedder | None = None,
) -> ToolRetrievalResult:
    """Return the top-K native tool schemas for a query."""
    if not tools:
        return ToolRetrievalResult([], [], {}, fallback_reason="no tools available")
    query = (query or "").strip()
    if not query:
        return ToolRetrievalResult(list(tools), [tool_name(t) for t in tools], {}, fallback_reason="empty query")

    cfg = config or {}
    try:
        top_k = int(cfg.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3
    if top_k <= 0:
        return ToolRetrievalResult(list(tools), [tool_name(t) for t in tools], {}, fallback_reason="top_k <= 0")

    embedder = embedder or embed_texts_openai_compatible
    index = load_or_build_index(tools, cfg, platform, embedder=embedder)
    query_vectors = embedder([query], cfg)
    if not query_vectors:
        raise ToolRetrievalError("query embedding was empty")
    query_vector = _normalize_vector(query_vectors[0])

    scores: list[tuple[float, str]] = []
    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        vector = entry.get("embedding")
        if name and isinstance(vector, list):
            scores.append((cosine_similarity(query_vector, vector), name))

    if not scores:
        raise ToolRetrievalError("no indexed tool scores available")

    scores.sort(key=lambda item: (-item[0], item[1]))
    selected_names = [name for _, name in scores[: min(top_k, len(scores))]]
    by_name = {tool_name(tool): tool for tool in tools}
    selected_tools = [by_name[name] for name in selected_names if name in by_name]
    if not selected_tools:
        return ToolRetrievalResult(list(tools), [tool_name(t) for t in tools], {}, fallback_reason="no selected tools found")
    return ToolRetrievalResult(
        selected_tools=selected_tools,
        selected_names=[tool_name(tool) for tool in selected_tools],
        scores={name: score for score, name in scores},
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
