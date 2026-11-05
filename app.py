from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from llm_service import generate_answer

st.set_page_config(
    page_title="AI Troubleshooting Assistant",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ AI Troubleshooting Assistant")
st.caption("Find step-by-step guidance for common technical issues using a sample knowledge base.")

with st.sidebar:
    st.header("Sample queries")
    sample_queries = [
        "The payment API is returning a 500 error",
        "My application cannot connect to the database",
        "Users are unable to log in",
        "The API is returning a 404 error",
        "The application is running slowly",
        "The service is unavailable",
    ]
    for query in sample_queries:
        if st.button(query, use_container_width=True):
            st.session_state["query_input"] = query

    st.divider()
    st.subheader("About")
    st.write("Answers are grounded in the local sample knowledge base.")
    st.write("Gemini selects MCP tools; the MCP server retrieves documented guidance.")
    st.write("Unknown issues should be acknowledged rather than guessed.")

st.subheader("Describe your technical issue")
query = st.text_area(
    "Issue description",
    value=st.session_state.get("query_input", ""),
    placeholder="For example: The payment API is returning a 500 error...",
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    analyze = st.button("Analyze issue", type="primary", use_container_width=True)
with col2:
    st.caption("Use a sample query or describe your own issue.")

if analyze:
    if not query.strip():
        st.warning("Enter a technical issue to begin.")
        st.stop()

    with st.spinner("Gemini is selecting MCP tools and analyzing the retrieved context..."):
        result = generate_answer(query)

    st.caption("Workflow: Gemini → MCP client → MCP server tools → Gemini response")

    st.subheader("MCP tool execution")
    tools_used = result.get("tools_used", [])
    if tools_used:
        for tool in tools_used:
            if tool.get("status") == "success":
                st.success(f'✓ {tool["name"]} — completed')
            else:
                st.warning(f'⚠ {tool["name"]} — failed')
    elif result.get("source") == "fallback":
        st.info("No successful Gemini-driven MCP tool calls were recorded.")
    else:
        st.info("Gemini did not invoke an MCP tool. Check the prompt and execution logs.")

    st.subheader("Troubleshooting guidance")
    st.markdown(result["answer"])

    if result.get("error") == "No relevant knowledge-base entry was identified.":
        st.warning("No relevant knowledge-base entries found.")

    if result.get("source") == "fallback":
        st.warning("AI generation was unavailable; a retrieval-based fallback is shown.")
        if result.get("error"):
            st.caption(f"Diagnostic category: {result['error'].split(':', 1)[0]}")

    sources = result.get("sources", [])
    if sources:
        st.subheader("Knowledge-base source")
        for entry in sources[:3]:
            title = entry.get("title", "Knowledge-base entry")
            entry_id = entry.get("id", "Unknown ID")
            score = entry.get("score")
            label = f"{entry_id} — {title}"
            if isinstance(score, (int, float)):
                label += f" (similarity: {score:.3f})"
            with st.expander(label, expanded=(entry == sources[0])):
                st.write("**Symptoms**")
                symptoms = entry.get("symptoms", [])
                st.write(", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms))
                st.write("**Diagnostic checks**")
                for item in entry.get("diagnostic_checks", []):
                    st.write(f"- {item}")
                st.write("**Known resolutions**")
                for item in entry.get("known_resolutions", []):
                    st.write(f"- {item}")
                st.write(f'**Escalation team:** {entry.get("escalation_team", "Not specified")}')
    else:
        st.info("No knowledge-base source was returned. Treat the response as insufficient until a relevant entry is retrieved.")

    st.caption(
        "Prototype limitation: similarity scores are heuristic and do not prove a diagnosis. "
        "Verify findings against logs, monitoring, and your organization's procedures."
    )

st.divider()
st.caption("Demo prototype • Uses sample knowledge only • Do not enter confidential production data")
