import json
import sys
import types

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent
from agent.tool_retrieval import ToolRetrievalResult

_PREPARED_INDEX = object()
_STABLE_RETRIEVAL_TOOL_NAMES = ["retrieve_tools", "call_retrieved_tool"]
_STABLE_RETRIEVAL_VALID_NAMES = {"retrieve_tools", "call_retrieved_tool"}


def _api_tool_names(agent) -> list[str]:
    return [tool["function"]["name"] for tool in agent._tools_for_api()]


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
    agent.session_id = "session"
    agent._memory_manager = None
    agent._session_db = None
    agent._context_engine_tool_names = set()
    agent._checkpoint_mgr = types.SimpleNamespace(enabled=False)
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
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES


def test_retrieve_tools_selects_current_api_tools_from_model_query(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        assert platform == "acp"
        assert prepared_index is _PREPARED_INDEX
        assert query == "fix test"
        return ToolRetrievalResult(
            selected_tools=[tools[1], tools[0]],
            selected_names=["terminal", "read_file"],
            scores={"terminal": 1.0, "read_file": 0.9},
        )

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES

    payload = json.loads(agent._retrieve_tools("fix test"))

    assert payload["success"] is True
    assert payload["exposed_tools"] == ["terminal", "read_file"]
    assert payload["retrieved_tools"] == ["terminal", "read_file"]
    assert [tool["name"] for tool in payload["tools"]] == ["terminal", "read_file"]
    assert payload["tools"][0]["score"] == 1.0
    assert payload["tools"][0]["description"] == "terminal tool"
    assert payload["tools"][0]["parameters"] == {"type": "object", "properties": {}}
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == ["terminal", "read_file"]
    assert agent.valid_tool_names == {"read_file", "terminal", "patch"}


def test_retrieve_tools_replaces_visible_tools_within_user_turn(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        if query == "run commands":
            return ToolRetrievalResult(
                selected_tools=[tools[1]],
                selected_names=["terminal"],
                scores={"terminal": 1.0},
            )
        if query == "edit files":
            return ToolRetrievalResult(
                selected_tools=[tools[2]],
                selected_names=["patch"],
                scores={"patch": 0.8},
            )
        raise AssertionError(query)

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    first = json.loads(agent._retrieve_tools("run commands"))
    second = json.loads(agent._retrieve_tools("edit files"))

    assert first["exposed_tools"] == ["terminal"]
    assert second["exposed_tools"] == ["patch"]
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == ["patch"]


def test_retrieve_tools_caps_current_visible_tools(monkeypatch):
    agent = _bare_agent(
        monkeypatch,
        "acp",
        {
            "tool_retrieval": {
                "enabled": True,
                "platforms": ["acp"],
                "top_k": 3,
                "max_visible_tools": 2,
            }
        },
    )

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        if query == "read run and edit":
            return ToolRetrievalResult(
                selected_tools=[tools[0], tools[1], tools[2]],
                selected_names=["read_file", "terminal", "patch"],
                scores={"read_file": 0.9, "terminal": 0.8, "patch": 0.7},
            )
        raise AssertionError(query)

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    payload = json.loads(agent._retrieve_tools("read run and edit"))

    assert payload["exposed_tools"] == ["read_file", "terminal"]
    assert [tool["name"] for tool in payload["tools"]] == ["read_file", "terminal"]
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._retrieved_tool_names == ["read_file", "terminal"]


def test_non_acp_prefilter_uses_full_catalog(monkeypatch):
    agent = _bare_agent(monkeypatch, "cli")

    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])

    assert _api_tool_names(agent) == [
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
    assert _api_tool_names(agent) == [
        "read_file",
        "terminal",
        "patch",
    ]
    assert agent._valid_tool_names_for_current_api_call() == {"read_file", "terminal", "patch"}


def test_cli_retrieve_tools_selects_tools_when_platform_enabled(monkeypatch):
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
        assert query == "fix test"
        return ToolRetrievalResult(
            selected_tools=[tools[2]],
            selected_names=["patch"],
            scores={"patch": 1.0},
        )

    agent._tool_retrieval_select_fn = fake_select
    agent._prepare_tool_prefilter_for_api_call("fix test", [{"role": "user", "content": "fix test"}])
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES

    payload = json.loads(agent._retrieve_tools("fix test"))

    assert payload["success"] is True
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == ["patch"]


def test_retrieve_tools_failure_returns_error_without_full_catalog_fallback(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fail_select(*_args, **_kwargs):
        raise RuntimeError("embedding service unavailable")

    agent._tool_retrieval_select_fn = fail_select

    payload = json.loads(agent._retrieve_tools("fix test"))

    assert payload["success"] is False
    assert "embedding service unavailable" in payload["error"]
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == []


def test_retrieve_tools_requires_query(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    payload = json.loads(agent._retrieve_tools("  "))

    assert payload["success"] is False
    assert payload["error"] == "query is required"
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == []


def test_invalid_hidden_tool_feedback_points_to_retrieve_tools(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    message = agent._invalid_tool_call_feedback("terminal", _STABLE_RETRIEVAL_VALID_NAMES)

    assert "currently hidden" in message
    assert "Call retrieve_tools first" in message
    assert "call it through call_retrieved_tool" in message


def test_invalid_unknown_tool_feedback_mentions_retrieve_tools_when_active(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    message = agent._invalid_tool_call_feedback("not_a_tool", _STABLE_RETRIEVAL_VALID_NAMES)

    assert "not currently available" in message
    assert "Available tools: call_retrieved_tool, retrieve_tools" in message
    assert "call retrieve_tools first" in message


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


def test_retrieve_tools_requires_prepared_index(monkeypatch):
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

    payload = json.loads(agent._retrieve_tools("fix test"))

    assert payload["success"] is False
    assert "index was not prepared at startup" in payload["error"]
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._retrieved_tool_names == []


def test_new_user_turn_resets_to_retrieval_only(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        return ToolRetrievalResult(
            selected_tools=[tools[1]],
            selected_names=["terminal"],
            scores={"terminal": 1.0},
        )

    agent._tool_retrieval_select_fn = fake_select
    json.loads(agent._retrieve_tools("run commands"))
    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._retrieved_tool_names == ["terminal"]

    agent._prepare_tool_prefilter_for_api_call("next task", [{"role": "user", "content": "next task"}])

    assert _api_tool_names(agent) == _STABLE_RETRIEVAL_TOOL_NAMES
    assert agent._valid_tool_names_for_current_api_call() == _STABLE_RETRIEVAL_VALID_NAMES
    assert agent._retrieved_tool_names == []


def test_call_retrieved_tool_dispatches_latest_retrieved_native_tool(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        return ToolRetrievalResult(
            selected_tools=[tools[1]],
            selected_names=["terminal"],
            scores={"terminal": 1.0},
        )

    calls = {}

    def fake_handle_function_call(function_name, function_args, task_id, **kwargs):
        calls["function_name"] = function_name
        calls["function_args"] = function_args
        calls["task_id"] = task_id
        calls["kwargs"] = kwargs
        return json.dumps({"ok": True})

    monkeypatch.setattr(run_agent, "handle_function_call", fake_handle_function_call)
    agent._tool_retrieval_select_fn = fake_select
    json.loads(agent._retrieve_tools("run commands"))

    result = json.loads(
        agent._call_retrieved_tool(
            "terminal",
            {"command": "pwd"},
            "task-1",
            tool_call_id="call-1",
        )
    )

    assert result == {"ok": True}
    assert calls["function_name"] == "terminal"
    assert calls["function_args"] == {"command": "pwd"}
    assert calls["task_id"] == "task-1"
    assert calls["kwargs"]["tool_call_id"] == "call-1"
    assert calls["kwargs"]["session_id"] == "session"
    assert set(calls["kwargs"]["enabled_tools"]) == {"terminal"}
    assert calls["kwargs"]["skip_pre_tool_call_hook"] is True


def test_call_retrieved_tool_rejects_tool_not_returned_by_latest_retrieval(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        return ToolRetrievalResult(
            selected_tools=[tools[1]],
            selected_names=["terminal"],
            scores={"terminal": 1.0},
        )

    def fail_handle_function_call(*_args, **_kwargs):
        raise AssertionError("unretrieved tool should not dispatch")

    monkeypatch.setattr(run_agent, "handle_function_call", fail_handle_function_call)
    agent._tool_retrieval_select_fn = fake_select
    json.loads(agent._retrieve_tools("run commands"))

    result = json.loads(
        agent._call_retrieved_tool(
            "patch",
            {"path": "app.py"},
            "task-1",
            tool_call_id="call-1",
        )
    )

    assert result["success"] is False
    assert "was not returned by the latest retrieve_tools call" in result["error"]
    assert result["retrieved_tools"] == ["terminal"]


def test_call_retrieved_tool_accepts_arguments_json_string(monkeypatch):
    agent = _bare_agent(monkeypatch, "acp")

    def fake_select(tools, query, config, platform=None, prepared_index=None):
        return ToolRetrievalResult(
            selected_tools=[tools[2]],
            selected_names=["patch"],
            scores={"patch": 1.0},
        )

    calls = {}

    def fake_handle_function_call(function_name, function_args, task_id, **kwargs):
        calls["function_name"] = function_name
        calls["function_args"] = function_args
        return "patched"

    monkeypatch.setattr(run_agent, "handle_function_call", fake_handle_function_call)
    agent._tool_retrieval_select_fn = fake_select
    json.loads(agent._retrieve_tools("edit files"))

    result = agent._call_retrieved_tool(
        "patch",
        '{"path": "app.py", "old": "a", "new": "b"}',
        "task-1",
    )

    assert result == "patched"
    assert calls == {
        "function_name": "patch",
        "function_args": {"path": "app.py", "old": "a", "new": "b"},
    }


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
