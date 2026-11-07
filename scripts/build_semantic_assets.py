#!/usr/bin/env python3
"""Download a frozen biomedical encoder and prepare or reuse ICD title vectors."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def catalogue_digest(records):
    payload = json.dumps(
        [[row["code"], row["title"]] for row in records],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode(texts, tokenizer, model, pooling, batch_size, max_length):
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**tokens).last_hidden_state
            if pooling == "cls":
                embeddings = hidden[:, 0]
            else:
                mask = tokens["attention_mask"].unsqueeze(-1)
                embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy())
    return np.concatenate(vectors).astype(np.float32)


def load_precomputed(folder, records, model_id, revision, pooling, max_length):
    """Reject stale or damaged public title matrices before downloading weights."""
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    expected = {"catalogue_sha256": catalogue_digest(records), "model_id": model_id,
                "model_revision": revision, "pooling": pooling, "max_length": max_length}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"precomputed {key} mismatch; use --rebuild to regenerate")
    path = folder / "title_embeddings.npy"
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if metadata.get("embeddings_sha256") != digest:
        raise ValueError("precomputed embedding checksum mismatch")
    vectors = np.load(path, mmap_mode="r", allow_pickle=False)
    if vectors.shape != (len(records), metadata.get("dimension")) or vectors.dtype != np.float32:
        raise ValueError("precomputed embedding shape/dtype mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("precomputed embeddings must be finite normalized vectors")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precomputed", type=Path, help="Directory containing catalogue vectors and metadata")
    parser.add_argument("--rebuild", action="store_true", help="Explicitly regenerate catalogue vectors")
    args = parser.parse_args()
    if args.rebuild and args.precomputed:
        parser.error("--rebuild and --precomputed cannot be combined")

    records = json.loads(args.catalogue.read_text(encoding="utf-8"))
    precomputed = args.precomputed
    metadata = None
    if precomputed is not None and not args.rebuild:
        metadata = load_precomputed(precomputed, records, args.model, args.revision, args.pooling, args.max_length)
        if precomputed.resolve() == args.output.resolve():
            raise ValueError("precomputed source must be separate from runtime output")
        print(f"Reusing precomputed catalogue embeddings from {precomputed}; preparing encoder weights.", flush=True)
    torch.set_num_threads(max(1, args.threads))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    if metadata is None:
        print(f"Encoding {len(records)} catalogue titles.", flush=True)
        vectors = encode(
            [row["title"] for row in records], tokenizer, model,
            args.pooling, args.batch_size, args.max_length,
        )
    elif model.config.hidden_size != metadata["dimension"]:
        raise ValueError("precomputed vectors do not match the encoder dimension")

    if args.output.exists():
        shutil.rmtree(args.output)
    model_dir = args.output / "model"
    model_dir.mkdir(parents=True)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    if metadata is not None:
        shutil.copy2(precomputed / "title_embeddings.npy", args.output / "title_embeddings.npy")
        shutil.copy2(precomputed / "metadata.json", args.output / "metadata.json")
        return
    np.save(args.output / "title_embeddings.npy", vectors, allow_pickle=False)
    metadata = {
        "model_id": args.model,
        "model_revision": args.revision,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "dimension": int(vectors.shape[1]),
        "catalogue_sha256": catalogue_digest(records),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    with (args.output / "title_embeddings.npy").open("rb") as source:
        metadata["embeddings_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
