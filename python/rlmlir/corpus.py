"""Manifest integrity and split protection at the measurement boundary."""
import hashlib
import json
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "configs/corpus.json"
DEVELOPMENT_SPLITS = ("train", "validation")
ALL_SPLITS = (*DEVELOPMENT_SPLITS, "heldout_shape", "heldout_structure")


class FrozenCorpus:
    """Verify identities once; never grant measurement access from a caller's label.

    Reading held-out shape metadata is not measuring its performance. There is
    deliberately no option here to enable held-out measurement.
    """
    def __init__(self, path=DEFAULT_CORPUS):
        path = Path(path)
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("frozen corpus hash mismatch")
        document = json.loads(raw)
        if document.get("schema_version") != 1 or not isinstance(document.get("workloads"), list):
            raise ValueError("unsupported corpus schema")
        self._identities = {}
        groups, shapes = {}, set()
        for item in document["workloads"]:
            if (set(item) != {"id", "family", "shape", "group", "split"}
                    or any(not isinstance(item[k], str) or not item[k] for k in ("id", "family", "group"))
                    or item["split"] not in ALL_SPLITS
                    or not isinstance(item["shape"], list) or not item["shape"]
                    or any(type(d) is not int or d < 0 for d in item["shape"])):
                raise ValueError("invalid corpus workload")
            if item["id"] in self._identities:
                raise ValueError("duplicate workload ID in frozen corpus")
            shape = (item["family"], tuple(item["shape"]))
            if shape in shapes:
                raise ValueError("duplicate family/shape in frozen corpus")
            shapes.add(shape)
            previous = groups.setdefault(item["group"], item["split"])
            if previous != item["split"]:
                raise ValueError("related shape groups crossed splits")
            # Serialized identities prevent a caller mutating a returned shape
            # list from also changing the authority used by the guard.
            self._identities[item["id"]] = json.dumps(item, sort_keys=True)

    def require_development(self, workload):
        identity = self._identities.get(workload.get("id"))
        if identity is None:
            raise ValueError("workload is absent from frozen corpus")
        canonical = json.loads(identity)
        if canonical["split"] not in DEVELOPMENT_SPLITS:
            raise ValueError("held-out measurements are sealed")
        if json.dumps(workload, sort_keys=True) != identity:
            raise ValueError("workload identity differs from frozen corpus")

    def development(self, split):
        if split not in DEVELOPMENT_SPLITS:
            raise ValueError("held-out performance is sealed until the final-method freeze")
        return [item for identity in self._identities.values()
                if (item := json.loads(identity))["split"] == split]


def load(path: Path, split: str):
    return FrozenCorpus(path).development(split)
