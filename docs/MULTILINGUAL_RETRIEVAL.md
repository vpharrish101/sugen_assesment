# Multilingual diagnosis focus: research and implementation

The multilingual query-focus addon was first tested locally on `tmp/ollama-rerank-download` and is now integrated in `src/coder/clinical_query.py`. It retains the pinned encoders and title vectors. The final English fallback uses the better measured original mixture (27/44), while non-English uses multilingual SapBERT. The extraction addon itself adds no inference call; the final pipeline can optionally rerank through Anthropic or Qwen.

## Research that informed the change

| Source | Relevant finding | Application here |
|---|---|---|
| [SapBERT, NAACL 2021](https://aclanthology.org/2021.naacl-main.334/) and [implementation](https://github.com/cambridgeltl/SapBERT) | The method aligns biomedical entity names. The repository's example embeds entity-name strings with a 25-token limit and CLS pooling. | Retrieve with the documented diagnosis phrase rather than an entire clinical narrative. Keep the official CLS pooling and model weights. |
| [Cross-lingual SapBERT / XL-BEL, ACL 2021](https://aclanthology.org/2021.acl-short.72/) | Cross-lingual biomedical entity linking transfers terminology knowledge, with substantial language-dependent performance differences. It is not validation of complete multilingual ICD coding. | Retain multilingual SapBERT for detected non-English notes, and measure each language rather than infer quality from the model name. |
| [ConText, 2009](https://doi.org/10.1016/j.jbi.2009.05.002) and [medspaCy](https://github.com/medspacy/medspacy) | Clinical NLP separates sections, experiencer, negation and uncertainty. medspaCy provides section/context components, but its published language table lists very few Spanish section rules, no Spanish ConText rules, and no ready Chinese resources. | Extend the small existing extractor with generic section, speaker and assertion cues. This is a narrow implementation inspired by those patterns, not a full ConText engine or automatic use of medspaCy's English rules. |
| Schwartz–Hearst (2003), implemented by [abbreviation-extraction](https://github.com/philgooch/abbreviation-extraction) | Parenthetical definitions in a document can resolve abbreviation/long-form pairs. | Add the small MIT-licensed `abbreviations==0.2.5` package. Expand only abbreviations whose definitions actually occur in the same note; do not add a global disease alias table. |

## Product change

`src/coder/clinical_query.py` now recognizes common Spanish diagnosis/assessment headings and Simplified Chinese headings, clinician-labelled diagnosis statements, and basic unlabelled clinician assertions. It stops at sentence/plan boundaries, excludes patient turns and marked history, and guards against questions and explicit negated assessments. Qualifying negation, such as an uncomplicated diagnosis, remains in the query. Where a selected abbreviation has an explicit definition in the note, the library resolves it (for example, `fibrilación auricular (FA)`); an undefined `FA` is left unchanged.

The implementation uses Python regexes for these few structural patterns, rather than installing spaCy and still having to supply Chinese rules. There are no disease names, ICD codes, note IDs, gold labels or manually added medical translations in the extraction rules. The English path remains intact: all 60 original English query strings and output predictions are unchanged in regression testing.

Build/install with the existing workflow (`docker compose build`, or install `requirements.txt` in the existing environment). The new dependency is one small Python package, not another model. In an environment without that optional import, lexical fallback remains usable but note-local abbreviation expansion is skipped; use the declared requirements for the tested behavior.

Unsupported languages/formats still fall back to the existing full-note query. The limited guards are not comprehensive clinical negation/NER, and this extractor must not be advertised as supporting every language. It neither infers a diagnosis nor manufactures absent organism/subtype information.

## Controlled evaluation

The same six supplied cases, exact reviewed Spanish/Chinese translations, unchanged gold and frozen model assets were used throughout. These visible translated cases informed development; they are not independent clinical validation. The 64-token encoder query limit and catalogue vectors were not changed by this implementation.

| Language (6 paired cases each) | Before top-1 / top-3 | After top-1 / top-3 | Gold in top 50 before → after | Truncated queries after |
|---|---|---|---|---|
| English | 2/6 / 4/6 | 2/6 / 4/6 | 6/6 → 6/6 | 0/6 |
| Spanish | 0/6 / 0/6 | 2/6 / 3/6 | 1/6 → 5/6 | 0/6 |
| Simplified Chinese | 0/6 / 0/6 | 0/6 / 2/6 | 1/6 → 4/6 | 0/6 |

The original weighted dual-encoder mixture produces the same paired counts as hard routing after the focus change. Official `score.py` scores are English 33.3/75, Spanish 29.2/75 and Chinese 8.3/75 on these six-case slices. The full 60-note hard-routed English run remains 26/60 top-one, 42/60 top-three (39.2/75), byte-identical to its previous output. Both implementations run offline with no semantic fallback or invalid/missing predictions.

New Spanish queries are 9–17 tokens and Chinese queries 8–17 tokens, replacing full notes of 104–287 tokens. The selected pneumonia translations still miss their gold in the top-50 pool, and the Chinese AF example still misses its exact gold. Phrase extraction therefore improves retrieval but does not establish reliable multilingual coding or resolve all contextual/specificity errors. A further mention-extraction/normalization approach must be evaluated if stronger performance is needed; disease-specific word removal should not be tuned to these six labels.

Tests cover headings, sentence/plan boundaries, questions, patient/clinician roles, negated/history statements, retained qualifiers and explicit-vs-undefined abbreviations. Runtime verification used the existing pinned model image with the changed source and verified resolver wheel; a fresh image build was not repeated for this local test. Full per-note queries, token lengths, model responses/rankings and source hashes are in `reports/multilingual_focus_evaluation.json`. Earlier reports are retained as controls.
