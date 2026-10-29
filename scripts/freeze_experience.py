"""Verify saved initial experience and pin its bytes; no model or native execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rlmlir.experience import load_pool, pool_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("manifest output must be new")
    # This verifies a historical pool against its own contract. Model fitting
    # independently checks that contract against the current compiler binary.
    expected = json.loads((args.runs[0] / "run.json").read_text())
    pool, sources = load_pool(args.runs, expected)
    manifest = pool_manifest(pool, sources, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fingerprint": manifest["fingerprint"], "trajectories": manifest["trajectories"],
                      "contexts": manifest["contexts"], "split_counts": manifest["split_counts"]}))


if __name__ == "__main__":
    main()
