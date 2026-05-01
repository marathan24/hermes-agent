import json
import sys
import types

import pytest

import agent.tool_retrieval as tool_retrieval
from agent.tool_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    ToolRetrievalError,
    build_tool_retrieval_query,
    index_artifact_paths,
    index_file_path,
    load_or_build_index,
    select_tools_for_query,
    tool_schema_hash,
    tool_schema_text,
)


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "required": list((properties or {}).keys())[:1],
                "properties": properties or {},
            },
        },
    }


def _rows(matrix):
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


class _FakeFaissIndexFlatIP:
    def __init__(self, dimensions):
        self.d = int(dimensions)
        self.vectors = []
        self.ntotal = 0

    def add(self, matrix):
        rows = _rows(matrix)
        for row in rows:
            if len(row) != self.d:
                raise ValueError("dimension mismatch")
        self.vectors.extend(rows)
        self.ntotal = len(self.vectors)

    def search(self, matrix, k):
        query_rows = _rows(matrix)
        all_scores = []
        all_indices = []
        for query in query_rows:
            scored = []
            for idx, vector in enumerate(self.vectors):
                score = sum(q * v for q, v in zip(query, vector))
                scored.append((score, idx))
            scored.sort(key=lambda item: item[0], reverse=True)
            top = scored[:k]
            all_scores.append([score for score, _idx in top])
            all_indices.append([idx for _score, idx in top])
        return all_scores, all_indices


class _FakeFaissModule:
    IndexFlatIP = _FakeFaissIndexFlatIP

    @staticmethod
    def write_index(index, path):
        data = {"d": index.d, "vectors": index.vectors}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @staticmethod
    def read_index(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        index = _FakeFaissIndexFlatIP(data["d"])
        index.add(data["vectors"])
        return index


@pytest.fixture
def fake_faiss(monkeypatch):
    module = _FakeFaissModule()
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module


def test_tool_schema_text_includes_parameter_names_and_descriptions():
    schema = _tool(
        "read_file",
        "Read a file from disk.",
        {
            "path": {"type": "string", "description": "Absolute file path"},
            "limit": {"type": "integer", "description": "Maximum lines"},
        },
    )

    text = tool_schema_text(schema)

    assert "name: read_file" in text
    assert "Read a file from disk." in text
    assert "parameter path required: string Absolute file path" in text
    assert "parameter limit: integer Maximum lines" in text


def test_index_artifact_paths_are_profile_safe(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    paths = index_artifact_paths(
        {"cache_dir": "cache/tool_retrieval", "model": DEFAULT_EMBEDDING_MODEL},
        "abc123",
        platform="acp",
    )

    expected_dir = (
        hermes_home
        / "cache"
        / "tool_retrieval"
        / "acp"
        / "sentence-transformers-all-minilm-l6-v2"
    )
    assert paths.index_path == expected_dir / "abc123.faiss"
    assert paths.metadata_path == expected_dir / "abc123.meta.json"
    assert index_file_path(
        {"cache_dir": "cache/tool_retrieval", "model": DEFAULT_EMBEDDING_MODEL},
        platform="acp",
        schema_hash="abc123",
    ) == paths.index_path


def test_sentence_transformer_model_is_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    tool_retrieval._MODEL_CACHE.clear()
    load_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            load_calls.append((model_name, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    config = {
        "model": "local/test-model",
        "device": "cpu",
        "model_cache_dir": "cache/tool_retrieval/models",
    }
    first = tool_retrieval._get_sentence_transformer_model(config)
    second = tool_retrieval._get_sentence_transformer_model(config)

    assert first is second
    assert len(load_calls) == 1
    assert load_calls[0][0] == "local/test-model"
    assert load_calls[0][1]["device"] == "cpu"


def test_load_or_build_index_saves_and_reuses_schema_embeddings(monkeypatch, tmp_path, fake_faiss):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    config = {"top_k": 2, "cache_dir": "cache/tool_retrieval", "model": "local/test-model"}
    tools = [_tool("terminal", "Run shell commands."), _tool("read_file", "Inspect source files.")]
    schema_embed_calls = 0

    def fake_embedder(texts, _config):
        nonlocal schema_embed_calls
        if all(text.startswith("name:") for text in texts):
            schema_embed_calls += 1
        return [[1.0, 0.0] if "terminal" in text else [0.0, 1.0] for text in texts]

    first = load_or_build_index(tools, config, platform="acp", embedder=fake_embedder)
    second = load_or_build_index(tools, config, platform="acp", embedder=fake_embedder)

    assert first.index_path == second.index_path
    assert first.metadata_path == second.metadata_path
    assert first.vector_dimensions == 2
    assert second.vector_dimensions == 2
    assert schema_embed_calls == 1
    assert first.index_path.exists()
    assert first.metadata_path.exists()


def test_select_with_prepared_index_embeds_only_query(monkeypatch, tmp_path, fake_faiss):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    config = {"top_k": 1, "cache_dir": "cache/tool_retrieval", "model": "local/test-model"}
    tools = [_tool("terminal", "Run shell commands."), _tool("web_search", "Search the web.")]
    calls = []

    def fake_embedder(texts, _config):
        calls.extend(texts)
        vectors = []
        for text in texts:
            low = text.lower()
            if "terminal" in low or "shell" in low:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    prepared = load_or_build_index(tools, config, platform="acp", embedder=fake_embedder)
    calls.clear()

    result = select_tools_for_query(
        tools,
        "run a shell command",
        config,
        platform="acp",
        embedder=fake_embedder,
        prepared_index=prepared,
    )

    assert result.selected_names == ["terminal"]
    assert calls == ["run a shell command"]


def test_stale_cache_rebuilds_for_schema_hash_model_platform_and_dimension(monkeypatch, tmp_path, fake_faiss):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    config = {"top_k": 1, "cache_dir": "cache/tool_retrieval", "model": "local/model-a"}
    tools_v1 = [_tool("read_file", "Read a file.")]
    tools_v2 = [_tool("read_file", "Read a file with pagination.")]
    schema_embed_calls = 0

    def fake_embedder(texts, _config):
        nonlocal schema_embed_calls
        if all(text.startswith("name:") for text in texts):
            schema_embed_calls += 1
        return [[1.0, 0.0] for _ in texts]

    first = load_or_build_index(tools_v1, config, platform="acp", embedder=fake_embedder)
    second = load_or_build_index(tools_v2, config, platform="acp", embedder=fake_embedder)
    model_changed = load_or_build_index(
        tools_v2,
        {**config, "model": "local/model-b"},
        platform="acp",
        embedder=fake_embedder,
    )
    platform_changed = load_or_build_index(tools_v2, config, platform="cli", embedder=fake_embedder)

    assert first.metadata["schema_hash"] == tool_schema_hash(tools_v1)
    assert second.metadata["schema_hash"] == tool_schema_hash(tools_v2)
    assert first.index_path != second.index_path
    assert second.index_path != model_changed.index_path
    assert second.index_path != platform_changed.index_path

    metadata = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    metadata["vector_dimensions"] = 3
    second.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    rebuilt = load_or_build_index(tools_v2, config, platform="acp", embedder=fake_embedder)

    assert rebuilt.vector_dimensions == 2
    assert schema_embed_calls == 5


def test_select_tools_for_query_validates_top_k(monkeypatch, tmp_path, fake_faiss):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with pytest.raises(ToolRetrievalError, match="top_k must be > 0"):
        select_tools_for_query([_tool("read_file", "Read a file.")], "read", {"top_k": 0})


def test_load_or_build_index_requires_tool_schemas(monkeypatch, tmp_path, fake_faiss):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with pytest.raises(ToolRetrievalError, match="no tools available"):
        load_or_build_index([], {"cache_dir": "cache/tool_retrieval"}, platform="acp")


def test_build_tool_retrieval_query_includes_recent_tool_context():
    query = build_tool_retrieval_query(
        "fix the failing test",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "terminal", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "content": "pytest failed in test_example"},
        ],
    )

    assert "Current user request:" in query
    assert "fix the failing test" in query
    assert "Recent assistant tool calls: terminal" in query
    assert "pytest failed in test_example" in query
