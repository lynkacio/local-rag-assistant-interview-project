import re
from collections.abc import Mapping, Sequence

from .generator import UNAVAILABLE_ANSWER


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate(
    result: Mapping[str, object],
    contexts: Sequence[Mapping[str, object]],
) -> bool:
    """Check the answer is verbatim in its cited retrieved source documents."""
    answer = result.get("answer")
    citations = result.get("citations")
    if not isinstance(answer, str) or not isinstance(citations, list):
        return False

    if answer == UNAVAILABLE_ANSWER:
        return not citations and not contexts
    if not answer.strip() or not citations:
        return False
    if any(not isinstance(citation, str) or not citation for citation in citations):
        return False
    if len(set(citations)) != len(citations):
        return False

    cited_contexts = [
        context for context in contexts if context.get("source") in citations
    ]
    if {context.get("source") for context in cited_contexts} != set(citations):
        return False
    source_text = _normalize(
        " ".join(
            str(context["text"])
            for context in cited_contexts
            if isinstance(context.get("text"), str)
        )
    )
    return _normalize(answer) in source_text
