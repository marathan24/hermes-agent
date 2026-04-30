import json
import sys
import types

from agent.tool_retrieval import (
    build_tool_retrieval_query,
    embed_texts_openai_compatible,
    index_file_path,
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


def test_index_file_path_is_profile_safe(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    path = index_file_path({"cache_dir": "cache/tool_retrieval"})

    assert path == hermes_home / "cache" / "tool_retrieval" / "index.json"


def test_select_tools_for_query_returns_top_k_with_fake_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    tools = [
        _tool("terminal", "Run shell commands."),
        _tool("read_file", "Inspect source files."),
        _tool("search_files", "Search repository text."),
        _tool("web_search", "Search the web."),
    ]

    vectors = {
        "terminal": [1.0, 0.0],
        "read_file": [0.95, 0.05],
        "search_files": [0.90, 0.10],
        "web_search": [0.0, 1.0],
    }

    def fake_embedder(texts, _config):
        result = []
        for text in texts:
            low = text.lower()
            if "inspect the repository" in low:
                result.append([1.0, 0.0])
                continue
            for name, vector in vectors.items():
                if name in low:
                    result.append(vector)
                    break
            else:
                result.append([0.0, 1.0])
        return result

    result = select_tools_for_query(
        tools,
        "inspect the repository and run the failing test",
        {"top_k": 3, "cache_dir": "cache/tool_retrieval"},
        platform="acp",
        embedder=fake_embedder,
    )

    assert result.fallback_reason is None
    assert result.selected_names == ["terminal", "read_file", "search_files"]


def test_schema_hash_change_rebuilds_cached_index(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    config = {"top_k": 1, "cache_dir": "cache/tool_retrieval"}
    tools_v1 = [_tool("read_file", "Read a file.")]
    tools_v2 = [_tool("read_file", "Read a file with pagination.")]
    build_calls = 0

    def fake_embedder(texts, _config):
        nonlocal build_calls
        if len(texts) > 1 or texts[0].startswith("name:"):
            build_calls += 1
        return [[1.0, 0.0] for _ in texts]

    select_tools_for_query(tools_v1, "read file", config, platform="acp", embedder=fake_embedder)
    first_metadata = json.loads(index_file_path(config).read_text(encoding="utf-8"))["metadata"]
    first_hash = first_metadata["schema_hash"]

    select_tools_for_query(tools_v2, "read file", config, platform="acp", embedder=fake_embedder)
    second_metadata = json.loads(index_file_path(config).read_text(encoding="utf-8"))["metadata"]
    second_hash = second_metadata["schema_hash"]

    assert first_hash == tool_schema_hash(tools_v1)
    assert second_hash == tool_schema_hash(tools_v2)
    assert first_hash != second_hash
    assert first_metadata["vector_dimensions"] == 2
    assert second_metadata["vector_dimensions"] == 2
    assert build_calls == 2


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


def test_embed_texts_openai_compatible_uses_openrouter_config(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def create(self, *, model, input):
            captured["model"] = model
            captured["input"] = input
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.embeddings = FakeEmbeddings()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    vectors = embed_texts_openai_compatible(
        ["tool schema"],
        {
            "model": "openai/text-embedding-3-small",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    )

    assert vectors == [[0.1, 0.2, 0.3]]
    assert captured["client_kwargs"] == {
        "api_key": "or-test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert captured["model"] == "openai/text-embedding-3-small"
    assert captured["input"] == ["tool schema"]
