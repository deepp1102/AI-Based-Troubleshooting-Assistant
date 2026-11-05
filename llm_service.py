"""Gemini-driven MCP tool calling for the troubleshooting assistant."""
import json
import os
from typing import Any

from mcp_client import call_tool
from retrieval import MIN_RELEVANCE_SCORE


def _unsupported_answer() -> str:
    return (
        "### No relevant knowledge-base information found\n"
        "The technical knowledge base does not contain relevant information for this query. "
        "I can't provide a diagnosis or troubleshooting steps without supporting knowledge-base evidence. "
        "Please describe a supported technical issue, including the affected service and the exact error message."
    )


TOOL_DECLARATIONS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the sample troubleshooting knowledge base for the user's issue. "
            "Call this first for every troubleshooting request. Returns ranked entries, "
            "similarity scores, diagnostic checks, known resolutions, and escalation teams."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "The user's technical issue."},
                "top_k": {
                    "type": "INTEGER",
                    "description": "Number of candidate entries to return (1 to 5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_issue_details",
        "description": "Retrieve full details for a knowledge-base entry by its ID, such as KB001.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "issue_id": {"type": "STRING", "description": "Knowledge-base ID."}
            },
            "required": ["issue_id"],
        },
    },
    {
        "name": "get_escalation_team",
        "description": "Retrieve the documented escalation team for a knowledge-base entry ID.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "issue_id": {"type": "STRING", "description": "Knowledge-base ID."}
            },
            "required": ["issue_id"],
        },
    },
]


def _fallback_answer(query: str, entries: list[dict[str, Any]]) -> str:
    """Transparent local fallback if Gemini is unavailable."""
    if not entries:
        return (
            "### Insufficient knowledge-base information\n"
            "No relevant knowledge-base entry is available, and AI generation is unavailable. "
            "I can't provide a diagnosis or troubleshooting steps without supporting evidence. "
            "Please provide a supported technical issue description with the affected service "
            "and exact error message."
        )
    entry = entries[0]
    lines = [
        f"### Issue summary\nYour query appears related to **{entry.get('title', 'the retrieved issue')}**.",
        "\n### Step-by-step diagnostic checks",
    ]
    lines.extend(
        f"{index}. {check}"
        for index, check in enumerate(entry.get("diagnostic_checks", []), start=1)
    )
    lines.append("\n### Documented resolutions (apply only after confirming the cause)")
    lines.extend(f"- {item}" for item in entry.get("known_resolutions", []))
    lines.append(
        f"\n### Escalation\nIf the checks do not resolve the issue, contact "
        f"**{entry.get('escalation_team', 'the relevant support team')}**."
    )
    lines.append(
        "\n> This is retrieval-based guidance from the sample knowledge base, not a confirmed diagnosis."
    )
    return "\n".join(lines)


def _as_genai_tool():
    """Build Gemini function declarations using the installed google-genai SDK."""
    from google.genai import types

    declarations = []
    for item in TOOL_DECLARATIONS:
        properties = {}
        for key, schema in item["parameters"]["properties"].items():
            field_type = {
                "STRING": types.Type.STRING,
                "INTEGER": types.Type.INTEGER,
            }[schema["type"]]
            properties[key] = types.Schema(
                type=field_type,
                description=schema.get("description", ""),
            )
        declarations.append(
            types.FunctionDeclaration(
                name=item["name"],
                description=item["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=properties,
                    required=item["parameters"].get("required", []),
                ),
            )
        )
    return types.Tool(function_declarations=declarations)


def generate_answer(query: str, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Let Gemini choose and call MCP tools, then generate a grounded final answer.
    `entries` remains optional for compatibility with older callers.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
    if not api_key:
        local_entries = [
            entry for entry in (entries or [])
            if isinstance(entry.get("score"), (int, float))
            and entry["score"] >= MIN_RELEVANCE_SCORE
        ]
        return {
            "answer": _fallback_answer(query, local_entries),
            "source": "fallback",
            "sources": local_entries,
            "tools_used": [],
            "error": "GEMINI_API_KEY is not configured.",
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        tool = _as_genai_tool()
        system_prompt = f"""
You are an L1/L2 technical troubleshooting assistant.
You have access to MCP tools exposed as function declarations.
For every new issue, you MUST call search_knowledge_base before giving troubleshooting advice.
Use get_issue_details and/or get_escalation_team when you need fuller details for a returned KB ID.
Do not claim to have called a tool unless you actually call it.
Use the knowledge-base results as the source of documented checks and resolutions.
Clearly distinguish the user's reported symptom from possible causes and confirmed causes.
A cause is confirmed only if the user supplies evidence that confirms it; a KB entry alone does not prove it.
Do not invent logs, metrics, system state, resolutions, or external service status.
Only treat a search result with a similarity score of at least {MIN_RELEVANCE_SCORE} as relevant.
If the search returns no relevant entries, state that the technical knowledge base does not contain relevant information, give no diagnosis or troubleshooting procedure, and ask for a supported technical issue description.
Never call get_issue_details or get_escalation_team before a relevant entry has been returned by search_knowledge_base. Only use IDs returned by that relevant search.
Give prioritized, numbered diagnostic steps, conditional resolutions, escalation guidance, and the KB IDs used.
Keep the answer practical and avoid repeating the entire KB entry verbatim.
"""
        contents: list[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{system_prompt}\n\nUser issue:\n{query}")],
            )
        ]
        tools_used: list[dict[str, str]] = []
        sources_by_id: dict[str, dict[str, Any]] = {}
        relevant_ids: set[str] = set()
        search_completed = False
        max_tool_rounds = 5

        for _ in range(max_tool_rounds):
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[tool],
                    temperature=0.2,
                ),
            )
            candidate = response.candidates[0] if response.candidates else None
            model_content = candidate.content if candidate else None
            if model_content is None:
                break

            contents.append(model_content)
            function_calls = [
                part.function_call
                for part in (model_content.parts or [])
                if getattr(part, "function_call", None) is not None
            ]
            if not function_calls:
                if not search_completed or not relevant_ids:
                    return {
                        "answer": _unsupported_answer() if search_completed else _fallback_answer(query, []),
                        "source": "unsupported",
                        "sources": [],
                        "tools_used": tools_used,
                        "error": "No relevant knowledge-base entry was identified.",
                    }
                answer = (getattr(response, "text", None) or "").strip()
                if answer:
                    return {
                        "answer": answer,
                        "source": "gemini",
                        "sources": list(sources_by_id.values()) or (entries or []),
                        "tools_used": tools_used,
                        "error": "",
                    }
                break

            function_response_parts = []
            for function_call in function_calls:
                name = function_call.name
                args = dict(function_call.args or {})
                issue_id = str(args.get("issue_id", ""))
                if name in {"get_issue_details", "get_escalation_team"} and issue_id not in relevant_ids:
                    payload = {"error": "A relevant knowledge-base search result is required before requesting entry details."}
                    tools_used.append({"name": name, "status": "failed"})
                    function_response_parts.append(
                        types.Part.from_function_response(name=name, response={"result": payload})
                    )
                    continue
                try:
                    result = call_tool(name, args)
                    tools_used.append({"name": name, "status": "success"})
                    if name == "search_knowledge_base":
                        search_completed = True
                        matches = result.get("matches", [])
                        if isinstance(matches, list):
                            for entry in matches:
                                if (isinstance(entry, dict) and entry.get("id")
                                        and isinstance(entry.get("score"), (int, float))
                                        and entry["score"] >= MIN_RELEVANCE_SCORE):
                                    relevant_ids.add(str(entry["id"]))
                                    sources_by_id[str(entry["id"])] = entry
                        if not relevant_ids:
                            return {
                                "answer": _unsupported_answer(),
                                "source": "unsupported",
                                "sources": [],
                                "tools_used": tools_used,
                                "error": "No relevant knowledge-base entry was identified.",
                            }
                    elif result.get("id"):
                        sources_by_id[result["id"]] = result
                    elif isinstance(result.get("matches"), list):
                        for entry in result["matches"]:
                            if isinstance(entry, dict) and entry.get("id"):
                                sources_by_id[entry["id"]] = entry
                    payload = result
                except Exception as exc:
                    print(f"MCP tool call failed ({name}): {type(exc).__name__}: {exc}", flush=True)
                    tools_used.append({"name": name, "status": "failed"})
                    payload = {"error": f"Tool {name} failed. Continue without assuming its result."}

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": payload},
                    )
                )

            contents.append(types.Content(role="user", parts=function_response_parts))

        raise RuntimeError("Gemini did not return a final text response within the tool-call limit.")

    except Exception as exc:
        # Keep the UI safe while leaving a useful diagnostic in the terminal.
        print(f"Gemini/MCP orchestration error: {type(exc).__name__}: {exc}", flush=True)
        fallback_entries = list(sources_by_id.values()) if "sources_by_id" in locals() else (entries or [])
        return {
            "answer": _fallback_answer(query, fallback_entries),
            "source": "fallback",
            "sources": fallback_entries,
            "tools_used": tools_used if "tools_used" in locals() else [],
            "error": f"{type(exc).__name__}: {exc}",
        }
