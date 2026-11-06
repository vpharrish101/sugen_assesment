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
        # Ambient transcripts can contain a patient's suspected diagnosis.
        # Only clinician turns are eligible when role labels are available.
        if re.match(r"(?i)^patient\s*:", segment):
            continue
        match = DIAGNOSIS_CUE.search(segment)
        if match:
            tail = segment[match.end():].strip(" :-")
            # A ruled-out candidate is not an assessment. If this removes every
            # cue, the full note remains available to retrieval below.
            if re.match(r"(?i)^(?:not\b|no\b|unlikely\b|excluded\b|ruled out\b)", tail):
                continue
            selected.append(tail or segment)
    return " ".join(selected) or text
