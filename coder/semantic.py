"""Local dense retrieval over precomputed ICD title embeddings."""

import hashlib
import json
from pathlib import Path


def catalogue_digest(records: list[dict]) -> str:
    """Hash the code/title sequence used to produce dense assets."""
    payload = json.dumps(
        [[row["code"], row["title"]] for row in records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DenseIndex:
    """Load a frozen encoder and matching normalized catalogue vectors."""

    def __init__(self, lexical_index, asset_dir: Path, threads: int = 4):
        try:
            import numpy as np
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("semantic dependencies are unavailable") from error

        asset_dir = Path(asset_dir)
        metadata = json.loads((asset_dir / "metadata.json").read_text(encoding="utf-8"))
        expected_digest = catalogue_digest(lexical_index.records)
        if metadata.get("catalogue_sha256") != expected_digest:
            raise ValueError("semantic assets do not match the supplied catalogue")
        vectors = np.load(asset_dir / "title_embeddings.npy", mmap_mode="r")
        expected_shape = (len(lexical_index.codes), metadata.get("dimension"))
        if vectors.shape != expected_shape:
            raise ValueError(
                f"semantic embedding shape {vectors.shape!r} does not match {expected_shape!r}"
            )

        torch.set_num_threads(max(1, threads))
        model_dir = asset_dir / "model"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModel.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.model.eval()
        self.np = np
        self.torch = torch
        self.codes = lexical_index.codes
        self.code_array = np.asarray(self.codes)
        self.vectors = vectors
        self.pooling = metadata["pooling"]
        self.max_length = metadata["max_length"]
        self.model_id = metadata["model_id"]

    def encode(self, texts: list[str]):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state
            if self.pooling == "cls":
                embeddings = hidden[:, 0]
            elif self.pooling == "mean":
                mask = tokens["attention_mask"].unsqueeze(-1)
                embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                raise ValueError(f"unsupported pooling {self.pooling!r}")
            embeddings = self.torch.nn.functional.normalize(embeddings, dim=1)
        return embeddings.cpu().numpy()

    def search(self, text: str, limit: int = 50) -> list[tuple[str, float]]:
        if limit <= 0:
            return []
        limit = min(limit, len(self.codes))
        query = self.encode([text])[0]
        scores = self.np.asarray(self.vectors @ query)
        candidates = self.np.argpartition(-scores, limit - 1)[:limit]
        ordered = candidates[
            self.np.lexsort((self.code_array[candidates], -scores[candidates]))
        ]
        return [(self.codes[index], float(scores[index])) for index in ordered]
