"""Copied tokenizer metadata must load under the pinned Transformers runtime."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, XLMRobertaTokenizerFast

from coder.semantic import DenseIndex, catalogue_digest


class TokenizerCompatibilityTests(unittest.TestCase):
    def test_copied_empty_list_metadata_loads_without_changing_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            model_dir = assets / "model"
            backend = Tokenizer(WordLevel(
                {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3,
                 "<mask>": 4, "example": 5}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            original = XLMRobertaTokenizerFast(tokenizer_object=backend)
            original.save_pretrained(model_dir)
            expected = original("example")["input_ids"]

            config_path = model_dir / "tokenizer_config.json"
            config = json.loads(config_path.read_text())
            config["extra_special_tokens"] = []
            config_path.write_text(json.dumps(config))
            # Reproduce the evaluator's exact error before exercising our loader.
            with self.assertRaisesRegex(AttributeError, "keys"):
                AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

            records = [{"code": "A", "title": "Example"}]
            lexical = Mock(records=records, codes=["A"])
            metadata = {"catalogue_sha256": catalogue_digest(records),
                        "dimension": 2, "pooling": "cls", "max_length": 64,
                        "model_id": "test-tokenizer"}
            (assets / "metadata.json").write_text(json.dumps(metadata))
            np.save(assets / "title_embeddings.npy",
                    np.array([[1, 0]], dtype=np.float32))
            with patch("transformers.AutoModel.from_pretrained", return_value=Mock()):
                index = DenseIndex(lexical, assets)
            self.assertEqual(index.tokenizer("example")["input_ids"], expected)


if __name__ == "__main__":
    unittest.main()
