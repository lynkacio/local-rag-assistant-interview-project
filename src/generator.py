import re


UNAVAILABLE_ANSWER = "Information is not available in the knowledge base."
_STOP_WORDS = {
    "a",
    "an",
    "are",
    "does",
    "for",
    "how",
    "is",
    "it",
    "of",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def _select_relevant_sentence(question: str, context: str) -> str:
    query_terms = {
        term
        for term in re.findall(r"\b[\w'-]+\b", question.casefold())
        if term not in _STOP_WORDS and len(term) > 1
    }
    if not query_terms:
        return ""

    context = re.sub(r"(?m)^\s{0,3}#{1,6}\s+.*$", "", context)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    ranked_sentences = []
    for position, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_terms = set(re.findall(r"\b[\w'-]+\b", sentence.casefold()))
        overlap = len(query_terms & sentence_terms)
        if overlap:
            ranked_sentences.append((overlap, -position, sentence))

    if not ranked_sentences:
        return ""
    return max(ranked_sentences)[2]


def generate_answer(
    contexts: list[dict[str, str | int | float]], question: str
) -> dict[str, str | list[str]]:
    """Return the most question-relevant sentence from the top retrieved chunk."""
    if not contexts:
        return {"answer": UNAVAILABLE_ANSWER, "citations": []}

    best_context = contexts[0]
    text = _select_relevant_sentence(question, str(best_context["text"]))
    source = str(best_context["source"])
    if not text or not source:
        return {"answer": UNAVAILABLE_ANSWER, "citations": []}
    return {"answer": text, "citations": [source]}
