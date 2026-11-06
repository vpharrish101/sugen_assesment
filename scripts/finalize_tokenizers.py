#!/usr/bin/env python3
"""Remove one-shot tokenizer migration flags from saved model metadata."""

import json
from pathlib import Path


for path in Path("assets/semantic").glob("*/model/tokenizer_config.json"):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.pop("fix_mistral_regex", None) is not None:
        path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
