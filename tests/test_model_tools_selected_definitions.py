from model_tools import get_tool_definitions_for_names


def _tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_get_tool_definitions_for_names_preserves_requested_order():
    tools = [_tool("read_file"), _tool("terminal"), _tool("patch")]

    selected = get_tool_definitions_for_names(tools, ["patch", "read_file"])

    assert [tool["function"]["name"] for tool in selected] == ["patch", "read_file"]


def test_get_tool_definitions_for_names_strips_unavailable_web_reference():
    tools = [
        _tool(
            "browser_navigate",
            "Navigate. For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
        ),
    ]

    selected = get_tool_definitions_for_names(tools, ["browser_navigate"])

    assert "web_search" not in selected[0]["function"]["description"]
