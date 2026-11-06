# MedGemma ICD-11 reranking experiment

Install Ollama and Python 3. Pull the model once:

```sh
ollama pull medgemma1.5:4b
```

From this directory, run a four-note technical check and then a clean 60-note comparison:

```sh
python run_ollama_experiment.py --limit 4 --output-prefix medgemma-schema-check
python run_ollama_experiment.py --output-prefix medgemma-schema
```

The candidate pool is frozen. Each request shows the clinical note, diagnosis-focus hint, and the available `ICD code: title` pairs. It asks MedGemma for **one** code, using an Ollama JSON schema that permits only the listed codes. The next request excludes the code already chosen. Three requests produce three distinct ranked codes. This removes the conflicting candidate numbers that caused MedGemma to mix indices and ICD codes. If the model still returns an invalid response, that note uses its saved retrieval baseline and the report counts a fallback; inference continues. Connection or Ollama server errors still stop the run so they can be diagnosed.

The default output prefix is `medgemma-schema`. Use `--resume --output-prefix medgemma-schema` after an interruption. Old output prefixes cannot be resumed with this method because the prompt and ranking procedure changed; keep the old files for comparison. The run writes predictions, trace, and results files with the selected prefix. `--cpu` forces CPU inference if GPU inference destabilizes the laptop.

This retains the 8,192-token context and 30-minute model lifetime from your fast run. It uses standard-library Python and no external inference calls. The supplied gold labels are read only after the 60-note run to compute top-one, top-three, and score out of 75. The SapBERT retrieval baseline is 27/60 top-one and 44/60 top-three (40.8/75). No MedGemma accuracy claim is possible until the full run completes.

Why this design: the [MedGemma 1.5 technical report](https://arxiv.org/abs/2604.05081) describes stronger EHR understanding, but does not establish ICD-11 coding specialization. The [MedGemma model metadata](https://huggingface.co/google/medgemma-1.5-4b-it) identifies a medical instruction model. [Ollama's API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-structured-outputs) supports a JSON schema in `format`. The local tests verify the request schema, three-code selection, invalid-response fallback, and rejection of old outputs. Live MedGemma inference must be checked on your laptop because the model is not installed in this cloud workspace.
