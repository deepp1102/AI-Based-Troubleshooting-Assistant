"""Small MCP client used by the Streamlit app to call the local stdio server."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent
SERVER_PATH = BASE_DIR / "mcp_server.py"


async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool '{tool_name}' returned an error")
        for item in getattr(result, "content", []):
            text = getattr(item, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"result": parsed}
                except json.JSONDecodeError:
                    return {"result": text}
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        raise RuntimeError(f"MCP tool '{tool_name}' returned no readable content")


def call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Synchronously call a local MCP tool from Streamlit."""
    return asyncio.run(_call_tool(tool_name, arguments))


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search through MCP and return ranked KB entries."""
    response = call_tool("search_knowledge_base", {"query": query, "top_k": top_k})
    matches = response.get("matches", [])
    return matches if isinstance(matches, list) else []
