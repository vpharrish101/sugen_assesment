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

# Section/assertion cues, not disease names. This small language layer follows
# the section + context approach used by clinical NLP tools such as medspaCy.
CLINICIAN = re.compile(r"(?i)^\s*(?:doctor|physician|m[eé]dic[oa]|医生|医师)\s*[:：]\s*")
PATIENT = re.compile(r"(?i)^\s*(?:patient|paciente|患者|病人|家属)\s*[:：]")
HISTORY = re.compile(r"(?i)^\s*(?:antecedentes|historia familiar|既往史|家族史)\s*[:：]")
MULTILINGUAL_HEADING = re.compile(
    r"(?i)(?<!\w)(?:impresi[oó]n\s+diagn[oó]stica|diagn[oó]stico(?:\s+(?:principal|actual|final))?|valoraci[oó]n)\s*[:：]\s*"
    r"|(?:初步诊断|临床诊断|最终诊断|诊断|评估)\s*[:：]\s*"
)
NEXT_SECTION = re.compile(
    r"(?i)(?<!\w)(?:plan|tratamiento|conducta|recomendaciones|pruebas|exploraci[oó]n)\s*[:：]"
    r"|(?:计划|治疗|处理|检查|建议)\s*[:：]"
)
MULTILINGUAL_SENTENCE = re.compile(r"(?<=[。！？])\s*|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ¿¡])")
ASSERTION = re.compile(
    r"(?i)^(?:(?:el|la)\s+m[eé]dic[oa]\s+(?:ha\s+diagnosticado|diagnostic[oó])|esto\s+es|se\s+trata\s+de|es)\s+"
    r"|^(?:医生诊断为|这是|这属于|诊断为)"
)
FOLLOWUP_CLAUSE = re.compile(r"(?i)\s+y\s+(?:recomend[oó]|prescribi[oó])\b|[，,]\s*(?:建议|计划|给予)")
NEGATED = re.compile(
    r"(?i)^(?:no\b(?!\s+complicad[oa]s?\b)|se\s+descarta\b|descartad[oa]\b|sin\s+evidencia\b|improbable\b)"
    r"|^(?:排除|不考虑|不是|未见|未发现|未证实|没有证据|无证据)"
)


def _expand_defined_abbreviations(focus: str, text: str) -> str:
    """Use only explicit note-local definitions, never a disease alias table."""
    short_forms = set(re.findall(r"(?<!\w)[A-Z][A-Z0-9-]{1,9}(?!\w)", focus))
    if not short_forms or '(' not in text:
        return focus
    try:
        from abbreviations import schwartz_hearst
    except ImportError:
        # Keep dependency-free lexical fallback usable outside the built image.
        return focus
    clinician_text = '\n'.join(line for line in text.splitlines() if not PATIENT.match(line))
    definitions = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=clinician_text)
    for short, long in definitions.items():
        if short in short_forms and long in clinician_text:
            focus = re.sub(r'(?<!\w)' + re.escape(short) + r'(?!\w)', lambda _: long, focus)
    return focus


def _multilingual_focus(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    has_speakers = any(CLINICIAN.match(line) or PATIENT.match(line) for line in lines)
    headings, assertions = [], []
    for line in lines:
        if PATIENT.match(line) or HISTORY.match(line):
            continue
        clinician = CLINICIAN.match(line)
        body = line[clinician.end():] if clinician else line
        matches = list(MULTILINGUAL_HEADING.finditer(body))
        for position, match in enumerate(matches):
            prefix = body[:match.start()]
            if re.search(r"(?i)(?:sin\s+|no\s+|既往|鉴别)$", prefix):
                continue
            end = matches[position + 1].start() if position + 1 < len(matches) else len(body)
            tail = NEXT_SECTION.split(body[match.end():end], maxsplit=1)[0]
            tail = FOLLOWUP_CLAUSE.split(MULTILINGUAL_SENTENCE.split(tail, maxsplit=1)[0], maxsplit=1)[0].strip()
            if tail and not NEGATED.match(tail) and '?' not in tail and '？' not in tail:
                headings.append(tail)
        if clinician or not has_speakers:
            for sentence in MULTILINGUAL_SENTENCE.split(body):
                sentence = sentence.strip()
                match = ASSERTION.match(sentence)
                if not match or '?' in sentence or '？' in sentence:
                    continue
                tail = FOLLOWUP_CLAUSE.split(sentence[match.end():], maxsplit=1)[0].strip()
                if tail and not NEGATED.match(tail) and not re.match(r"(?i)^posible\b", tail):
                    assertions.append(tail)
    return ' '.join(dict.fromkeys(headings or assertions))


def diagnosis_focus(text: str) -> str:
    """Return cue-led assessment text, or the original note when no cue exists.

    The rule recognizes documentation structure and generic diagnostic language.
    It contains no disease names, code prefixes, or dev-note identifiers.
    """
    multilingual = _multilingual_focus(text)
    if multilingual:
        return _expand_defined_abbreviations(multilingual, text)
    selected = []
    for segment in (part.strip() for part in SENTENCE.split(text)):
        if not segment:
            continue
        # Ambient transcripts can contain a patient's suspected diagnosis.
        # Only clinician turns are eligible when role labels are available.
        if PATIENT.match(segment):
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
