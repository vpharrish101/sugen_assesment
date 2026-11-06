"""Source-checkout entry point: python run.py INPUT.jsonl OUTPUT.jsonl."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from coder.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
