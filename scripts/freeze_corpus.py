"""Freeze shape groups before training; pilot groups can appear only in training."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def group(family, shape):
    # Neighbors around a 16-element anchor belong together, rather than splitting +/-1 variants.
    return family + ":" + ",".join(str((d + 8) // 16) for d in shape)


def main():
    destination = ROOT / "configs/corpus.json"
    if destination.exists():
        raise SystemExit("corpus is already frozen; revisions need an explicit new protocol")
    rng = random.Random(503)
    pilot = json.loads((ROOT / "configs/pilot.json").read_text())
    entries, used_shapes, used_groups = [], set(), set()
    def add(family, shape, split):
        shape_key = (family, tuple(shape))
        group_id = group(family, shape)
        if shape_key in used_shapes or group_id in used_groups:
            return False
        used_shapes.add(shape_key)
        used_groups.add(group_id)
        identity = hashlib.sha256(json.dumps(shape_key).encode()).hexdigest()[:16]
        entries.append({"id": identity, "family": family, "shape": shape, "group": group_id, "split": split})
        return True
    for family in ("matmul", "matvec", "row_reduce", "elementwise"):
        for shape in pilot.get(family + "_shapes", []):
            if not add(family, shape, "train"):
                raise RuntimeError("pilot shape groups overlap; resolve before freezing")
        for split, target in [("train",32), ("validation",8), ("heldout_shape",8)]:
            attempts = 0
            while sum(x["family"] == family and x["split"] == split for x in entries) < target:
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError("insufficient independent shape groups")
                limits = [8,8,8] if family == "matmul" else [24,16]
                shape = [16*rng.randint(1,limit) + rng.randint(-3,3) for limit in limits]
                add(family, shape, split)
    for family in ("stencil", "conv1d"):
        while sum(x["family"] == family for x in entries) < 16:
            shape = ([16*rng.randint(1,6)+rng.randint(-3,3),16*rng.randint(1,6)+rng.randint(-3,3)]
                     if family == "stencil" else [16*rng.randint(1,32)+rng.randint(-3,3), rng.choice([3,5,9,17])])
            add(family, shape, "heldout_structure")
    document = {"schema_version": 1, "seed": 503, "group_rule": "family plus nearest16-element anchor per dimension",
                "heldout_performance": "sealed until all core and advanced methods are frozen", "workloads": entries}
    data = (json.dumps(document, indent=2) + "\n").encode()
    destination.write_bytes(data)
    destination.with_suffix(".sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    print({split: sum(e["split"] == split for e in entries) for split in ("train","validation","heldout_shape","heldout_structure")})


if __name__ == "__main__":
    main()
