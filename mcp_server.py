"""Local MCP server exposing the troubleshooting knowledge base as tools.

Run with: python mcp_server.py
The server uses stdio transport; stdout is reserved for MCP protocol messages.
"""
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from retrieval import load_knowledge_base, retrieve_entries

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"
server = FastMCP("AI Troubleshooting Knowledge Base")


def _load_kb() -> list[dict[str, Any]]:
    return load_knowledge_base(str(KB_PATH))


@server.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the sample troubleshooting knowledge base for a technical issue.

    Returns ranked entries with similarity scores, symptoms, diagnostic checks,
    known resolutions, escalation team, and source identifiers. Scores are
    heuristic and do not prove the cause of an incident.
    """
    if not query or not query.strip():
        return json.dumps({"matches": [], "message": "Please provide an issue description."})
    safe_top_k = max(1, min(int(top_k), 5))
    matches = retrieve_entries(query, _load_kb(), top_k=safe_top_k)
    return json.dumps({"matches": matches}, ensure_ascii=False)


@server.tool()
def get_issue_details(issue_id: str) -> str:
    """Get the full symptoms, diagnostic checks, resolutions, and source for a KB issue ID."""
    for entry in _load_kb():
        if str(entry.get("id", "")).casefold() == issue_id.strip().casefold():
            return json.dumps(entry, ensure_ascii=False)
    return json.dumps({"error": f"No knowledge-base issue found with ID '{issue_id}'."})


@server.tool()
def get_escalation_team(issue_id: str) -> str:
    """Return the documented escalation team for a knowledge-base issue ID."""
    for entry in _load_kb():
        if str(entry.get("id", "")).casefold() == issue_id.strip().casefold():
            return json.dumps({
                "id": entry.get("id"),
                "title": entry.get("title"),
                "escalation_team": entry.get("escalation_team", "Not specified"),
                "source": entry.get("source", entry.get("id")),
            }, ensure_ascii=False)
    return json.dumps({"error": f"No knowledge-base issue found with ID '{issue_id}'."})


if __name__ == "__main__":
    server.run(transport="stdio")
