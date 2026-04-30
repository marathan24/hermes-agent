import json
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


def test_tools_prewarm_command_builds_index_json(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / "tool-cache"
    config = {
        "tool_retrieval": {
            "enabled": True,
            "platforms": ["cli"],
            "top_k": 3,
            "model": "openai/text-embedding-3-small",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "cache_dir": str(cache_dir),
            "index_filename": "index.json",
        }
    }
    tools = [_tool("terminal"), _tool("read_file")]
    captured = {}

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
        return {
            "metadata": {
                "embedding_model": actual_config["model"],
                "platform": platform,
                "tool_count": len(actual_tools),
                "vector_dimensions": 1536,
            },
            "entries": [],
        }

    monkeypatch.setattr(tool_retrieval, "load_or_build_index", fake_load_or_build_index)

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
    assert payload["metadata"]["embedding_model"] == "openai/text-embedding-3-small"
    assert payload["metadata"]["vector_dimensions"] == 1536
    assert payload["index_path"] == str(cache_dir / "index.json")
    assert captured == {
        "tools": tools,
        "config": config["tool_retrieval"],
        "platform": "cli",
    }


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
