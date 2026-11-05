# AI Troubleshooting Assistant

A small prototype for the Technology Operations Intern practical assignment. It retrieves relevant entries from a local sample knowledge base, presents step-by-step troubleshooting guidance, shows source entries, and handles unsupported queries conservatively.

## Features

- Streamlit chat/search-style interface.
- Sample JSON knowledge base with 15 technical issues.
- TF-IDF and cosine-similarity retrieval.
- A local MCP server exposing knowledge-base search, issue-detail, and escalation-team tools.
- An MCP client used by the Streamlit app to call the server over stdio, with a direct-retrieval fallback if MCP is unavailable.
- Optional Gemini API response generation using retrieved knowledge.
- Local fallback response if the API key is missing or the API is unavailable.
- Source entry display, relevance score, and escalation-team guidance.
- Basic unknown-query handling using a configurable retrieval threshold.

## Requirements

- Python 3.10 or newer for the application; the pinned MCP SDK v1 line supports the project setup used here. Python 3.11+ is recommended.
- Internet access only if you want to use Gemini.
- A Gemini API key is optional. Free-tier availability, quotas, and model access depend on your account and Google's current terms.

## Setup (macOS / Windows)

### 1. Open the project folder

Open this directory in VS Code or your preferred IDE.

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the optional Gemini API key

Copy `.env.example` to `.env` and add your key:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

The application automatically loads `.env` using `python-dotenv`. Alternatively, you can export the variables in your terminal before starting the app.

**Never commit `.env` or expose your API key.** Do not enter confidential production logs or personal data into an external LLM.

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will show a local URL in the terminal, usually `http://localhost:8501`.

## How it works

1. The user describes a technical issue in plain language.
2. `app.py` calls `mcp_client.py`, which launches `mcp_server.py` as a local stdio subprocess.
3. The MCP server exposes three tools: `search_knowledge_base`, `get_issue_details`, and `get_escalation_team`.
4. `search_knowledge_base` delegates ranking to `retrieval.py`, which uses TF-IDF and cosine similarity over the local JSON knowledge base.
5. The app applies a relevance threshold before generating an answer. If the MCP call fails, it falls back to direct local retrieval and displays a warning.
6. `llm_service.py` sends the question and retrieved entries to Gemini, if configured. If Gemini is unavailable, the app uses its local fallback response.
7. The UI displays the troubleshooting guidance and supporting knowledge-base source entries.

### MCP design

This is a local, tool-oriented MCP implementation—not a connection to Pine Labs' payment MCP server. MCP standardizes how the application exposes and calls tools; it does not itself improve retrieval accuracy or verify a diagnosis. The server uses stdio, so it is launched by the local MCP client and does not open a network port.

Tools exposed by the server:
- `search_knowledge_base(query, top_k)` — ranked matching entries.
- `get_issue_details(issue_id)` — full details for a known entry ID.
- `get_escalation_team(issue_id)` — documented escalation team for a known entry ID.

This is a lightweight retrieval-augmented generation (RAG)-style prototype. It does not train or fine-tune an LLM.

## Testing checklist

Try:
- `The payment API is returning a 500 error`
- `My app cannot connect to the database`
- `Users are unable to log in`
- `Why am I getting HTTP 429?`
- `The application became slow after a deployment`
- An unrelated query such as `How do I bake sourdough bread?`

Check that relevant entries are displayed for supported issues and unrelated queries are not confidently presented as solved. The retrieval threshold is a prototype heuristic and must be tuned through testing; it is not a guarantee of relevance.

## Assumptions and limitations

- All knowledge-base records are illustrative sample data, not production runbooks.
- Similarity scores measure text similarity, not factual correctness or causal probability.
- A matching entry does not prove the root cause.
- The LLM can still make mistakes; verify its output and source grounding.
- The sample fallback uses the top retrieved entry and is not a confirmed diagnosis.
- This is a demo prototype, not a production incident-management system.
- The initial implementation does not retain conversation history or connect to production systems.

## Suggested 10-minute demo

1. (1 min) Explain the problem and intended users.
2. (2 min) Show the knowledge-base structure and retrieval approach.
3. (3 min) Demonstrate a payment API 500 error and source entry.
4. (2 min) Demonstrate a different supported issue.
5. (1 min) Demonstrate an unsupported query.
6. (1 min) Explain limitations, testing, and future improvements.

## Project structure

```text
ai-troubleshooting-assistant/
├── app.py
├── retrieval.py
├── llm_service.py
├── mcp_server.py
├── mcp_client.py
├── knowledge_base.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
