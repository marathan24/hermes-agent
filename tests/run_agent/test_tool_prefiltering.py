import sys
import types

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.tool_retrieval import ToolRetrievalResult

_PREPARED_INDEX = object()


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _bare_agent(
    monkeypatch,
    platform: str = "acp",
    config: dict | None = None,
    preload_calls: list | None = None,
):
    import agent.tool_retrieval as tool_retrieval

    def fake_load_or_build_index(tools, retrieval_config, platform=None):
        return _PREPARED_INDEX

    def fake_preload_embedding_model(retrieval_config):
        if preload_calls is not None:
            preload_calls.append(retrieval_config)

    monkeypatch.setattr(tool_retrieval, "load_or_build_index", fake_load_or_build_index)
    monkeypatch.setattr(tool_retrieval, "preload_embedding_model", fake_preload_embedding_model)
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = platform
    agent.tools = [_tool("read_file"), _tool("terminal"), _tool("patch")]
    agent.valid_tool_names = {"read_file", "terminal", "patch"}
    agent.verbose_logging = False
    if config is None:
        config = {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["acp"],
                "top_k": 3,
            }
        }
    agent._configure_tool_retrieval(config)
    return agent


def test_configure_tool_retrieval_prepares_index_at_startup(monkeypatch):
    preload_calls = []
    agent = _bare_agent(monkeypatch, "acp", preload_calls=preload_calls)

    assert agent._tool_retrieval_enabled is True
    assert agent._tool_retrieval_index is _PREPARED_INDEX
    assert preload_calls == [agent._tool_retrieval_config]


def test_acp_prefilter_selects_current_api_tools(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        assert platform == "acp"
        assert prepared_index is _PREPARED_INDEX
        assert "fix test" in query
        return ToolRetrievalResult(
            selected_tools=[tools[1], tools[0]],
            selected_names=["terminal", "read_file"],
            scores={"terminal": 1.0, "read_file": 0.9},
        )

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    assert [tool["function"]["name"] for tool in agent._tools_for_api()] == ["terminal", "read_file"]
    assert agent._valid_tool_names_for_current_api_call() == {"terminal", "read_file"}
    assert agent.valid_tool_names == {"read_file", "terminal", "patch"}


def test_non_acp_prefilter_uses_full_catalog(monkeypatch):
    agent = _bare_agent(monkeypatch, "cli")

    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    assert [tool["function"]["name"] for tool in agent._tools_for_api()] == [
        "read_file",
        "terminal",
        "patch",
    ]
    assert agent._valid_tool_names_for_current_api_call() == {"read_file", "terminal", "patch"}


def test_disable_tool_retrieval_uses_full_catalog_without_embedding_index(monkeypatch):
    import agent.tool_retrieval as tool_retrieval

    def fail_load_or_build_index(*_args, **_kwargs):
        raise AssertionError("tool retrieval index should not be built")

    def fail_preload_embedding_model(*_args, **_kwargs):
        raise AssertionError("embedding model should not be preloaded")

    monkeypatch.setattr(tool_retrieval, "load_or_build_index", fail_load_or_build_index)
    monkeypatch.setattr(tool_retrieval, "preload_embedding_model", fail_preload_embedding_model)
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = "cli"
    agent.disable_tool_retrieval = True
    agent.tools = [_tool("read_file"), _tool("terminal"), _tool("patch")]
    agent.valid_tool_names = {"read_file", "terminal", "patch"}
    agent.verbose_logging = False

    agent._configure_tool_retrieval(
        {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["cli"],
                "top_k": 3,
            }
        }
    )
    agent._prepare_tool_prefilter_for_api_call(
        "fix test",
        [{"role": "user", "content": "fix test"}],
    )

    assert agent._tool_retrieval_enabled is False
    assert agent._tool_retrieval_index is None
    assert [tool["function"]["name"] for tool in agent._tools_for_api()] == [
        "read_file",
        "terminal",
        "patch",
    ]
    assert agent._valid_tool_names_for_current_api_call() == {"read_file", "terminal", "patch"}


def test_cli_prefilter_selects_tools_when_platform_enabled(monkeypatch):
    agent = _bare_agent(monkeypatch, "cli")
    agent._configure_tool_retrieval(
        {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["cli"],
                "top_k": 3,
            }
        }
    )

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        assert platform == "cli"
        assert prepared_index is _PREPARED_INDEX
        assert "fix test" in query
        return ToolRetrievalResult(
            selected_tools=[tools[2]],
            selected_names=["patch"],
            scores={"patch": 1.0},
        )

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    assert [tool["function"]["name"] for tool in agent._tools_for_api()] == ["patch"]
    assert agent._valid_tool_names_for_current_api_call() == {"patch"}


def test_prefilter_failure_raises_without_full_catalog_fallback(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fail_select(*_args, **_kwargs):
        raise RuntimeError("embedding service unavailable")

    agent._tool_retrieval_select_fn = fail_select

    with pytest.raises(RuntimeError, match="tool retrieval failed.*embedding service unavailable"):
        agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])


def test_configure_tool_retrieval_fails_closed_when_startup_index_build_fails(monkeypatch):
    import agent.tool_retrieval as tool_retrieval

    def fail_load_or_build_index(*_args, **_kwargs):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(tool_retrieval, "load_or_build_index", fail_load_or_build_index)
    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent.platform = "acp"
    agent.tools = [_tool("read_file")]
    agent.valid_tool_names = {"read_file"}

    with pytest.raises(RuntimeError, match="tool retrieval initialization failed.*model download failed"):
        agent._configure_tool_retrieval(
            {
                "tool_retrieval": {
                    "enabled": True,
                    "platforms": ["acp"],
                    "top_k": 3,
                }
            }
        )


def test_prefilter_requires_prepared_index(monkeypatch):
    agent = _bare_agent(
        monkeypatch,
        "cli",
        {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["cli"],
                "top_k": 3,
            }
        },
    )
    agent._tool_retrieval_index = None

    with pytest.raises(RuntimeError, match="index was not prepared at startup"):
        agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])


def test_tool_name_repair_only_uses_current_api_tool_names(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")
    agent._current_api_tools = [_tool("read_file")]
    agent._current_valid_tool_names = {"read_file"}

    assert agent._repair_tool_call("ReadFile") == "read_file"
    assert agent._repair_tool_call("TerminalTool") is None


class _FakeTransport:
    def build_kwargs(self, **kwargs):
        return kwargs


def test_build_api_kwargs_sends_current_api_tools(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")
    selected = [_tool("terminal")]
    agent._current_api_tools = selected
    agent.api_mode = "chat_completions"
    agent.model = "test-model"
    agent.provider = "test"
    agent.base_url = "https://example.invalid/v1"
    agent._base_url_lower = agent.base_url
    agent.providers_allowed = []
    agent.providers_ignored = []
    agent.providers_order = []
    agent.provider_sort = ""
    agent.provider_require_parameters = False
    agent.provider_data_collection = None
    agent.session_id = "session"
    agent.max_tokens = None
    agent.reasoning_config = None
    agent.request_overrides = {}
    agent._ollama_num_ctx = None
    agent._ephemeral_max_output_tokens = None
    agent._get_transport = lambda: _FakeTransport()
    agent._is_qwen_portal = lambda: False
    agent._is_openrouter_url = lambda: False
    agent._prepare_messages_for_non_vision_model = lambda messages: messages
    agent._resolved_api_call_timeout = lambda: None
    agent._max_tokens_param = lambda *_args, **_kwargs: "max_tokens"
    agent._supports_reasoning_extra_body = lambda: False
    agent._github_models_reasoning_extra_body = lambda: None

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    assert kwargs["tools"] is selected
