"""Extract diagnosis-focused text without disease-specific rules."""

import re


SENTENCE = re.compile(r"[\n]|(?<=[.!?])\s+")
DIAGNOSIS_CUE = re.compile(
    r"(?i)"
    r"\b(?:a|assessment|imp|impression)\s*:\s*"
    r"|\b(?:dx|diagnosis)\s*:?\s+"
    r"|\b(?:"
    r"diagnosed?\s+(?:with|as)|this is|it(?:\s+is|'s)|what you have is|"
    r"these values show|so (?:this|it) is|assessment is|impression is"
    r")\b"
)


def diagnosis_focus(text: str) -> str:
    """Return cue-led assessment text, or the original note when no cue exists.

    The rule recognizes documentation structure and generic diagnostic language.
    It contains no disease names, code prefixes, or dev-note identifiers.
    """
    selected = []
    for segment in (part.strip() for part in SENTENCE.split(text)):
        if not segment:
            continue
        match = DIAGNOSIS_CUE.search(segment)
        if match:
            tail = segment[match.end():].strip(" :-")
            selected.append(tail or segment)
    return " ".join(selected) or text
