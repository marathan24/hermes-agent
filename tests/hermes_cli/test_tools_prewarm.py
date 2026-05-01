import json
import types
from argparse import Namespace

from hermes_cli import tools_config


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_tools_prewarm_command_builds_faiss_artifacts(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / "tool-cache"
    config = {
        "tool_retrieval": {
            "enabled": True,
            "platforms": ["cli"],
            "top_k": 3,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "normalize_embeddings": True,
            "index_backend": "faiss",
            "index_type": "flat_ip",
            "cache_dir": str(cache_dir),
            "model_cache_dir": str(cache_dir / "models"),
        }
    }
    tools = [_tool("terminal"), _tool("read_file")]
    captured = {}
    preload_calls = []

    monkeypatch.setattr(tools_config, "load_config", lambda: config)

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda enabled_toolsets, quiet_mode: tools,
    )

    import agent.tool_retrieval as tool_retrieval

    def fake_load_or_build_index(actual_tools, actual_config, platform=None):
        captured["tools"] = actual_tools
        captured["config"] = actual_config
        captured["platform"] = platform
        schema_hash = tool_retrieval.tool_schema_hash(actual_tools)
        paths = tool_retrieval.index_artifact_paths(actual_config, schema_hash, platform)
        metadata = {
            "model": actual_config["model"],
            "embedding_model": actual_config["model"],
            "platform": platform,
            "tool_count": len(actual_tools),
            "vector_dimensions": 384,
            "schema_hash": schema_hash,
        }
        return types.SimpleNamespace(
            metadata=metadata,
            vector_dimensions=384,
            index_path=paths.index_path,
            metadata_path=paths.metadata_path,
        )

    monkeypatch.setattr(tool_retrieval, "load_or_build_index", fake_load_or_build_index)
    monkeypatch.setattr(tool_retrieval, "preload_embedding_model", lambda cfg: preload_calls.append(cfg))

    tools_config.tools_prewarm_command(
        Namespace(
            platform="cli",
            toolsets="hermes-cli",
            json=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["platform"] == "cli"
    assert payload["toolsets"] == ["hermes-cli"]
    assert payload["tool_count"] == 2
    assert payload["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert payload["vector_dimensions"] == 384
    assert payload["metadata"]["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert payload["metadata"]["vector_dimensions"] == 384
    assert payload["schema_hash"] == tool_retrieval.tool_schema_hash(tools)
    assert payload["index_path"].endswith(f"{payload['schema_hash']}.faiss")
    assert payload["metadata_path"].endswith(f"{payload['schema_hash']}.meta.json")
    assert str(cache_dir / "cli" / "sentence-transformers-all-minilm-l6-v2") in payload["index_path"]
    assert captured == {
        "tools": tools,
        "config": config["tool_retrieval"],
        "platform": "cli",
    }
    assert preload_calls == [config["tool_retrieval"]]


def test_tools_prewarm_command_requires_enabled_platform(monkeypatch, capsys):
    monkeypatch.setattr(
        tools_config,
        "load_config",
        lambda: {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["acp"],
            }
        },
    )

    try:
        tools_config.tools_prewarm_command(
            Namespace(
                platform="cli",
                toolsets="hermes-cli",
                json=True,
            )
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("prewarm should fail when tool_retrieval excludes cli")

    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert "not enabled for platform 'cli'" in payload["error"]
