"""Simple TF-IDF retrieval for the local sample knowledge base."""
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Scores below this heuristic cutoff tend to be generic word overlap rather
# than a useful match. Keep it shared by MCP retrieval and orchestration.
MIN_RELEVANCE_SCORE = 0.1


def load_knowledge_base(path: str = "knowledge_base.json") -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {file_path.resolve()}. "
            "Run the application from the project root."
        )
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("knowledge_base.json must contain a JSON list.")
    return data


def _entry_text(entry: dict[str, Any]) -> str:
    fields = [
        entry.get("title", ""),
        entry.get("description", ""),
        " ".join(entry.get("symptoms", [])),
        " ".join(entry.get("diagnostic_checks", [])),
        " ".join(entry.get("known_resolutions", [])),
    ]
    return " ".join(str(value) for value in fields if value)


def retrieve_entries(query: str, knowledge_base: list[dict[str, Any]], top_k: int = 3):
    """Return entries ranked by cosine similarity to the user's query."""
    if not query.strip() or not knowledge_base:
        return []

    documents = [_entry_text(entry) for entry in knowledge_base]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    ranked = sorted(
        zip(knowledge_base, scores),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        {**entry, "score": float(score)}
        for entry, score in ranked
        if score >= MIN_RELEVANCE_SCORE
    ]
