"""Fit the non-RL cost baseline on the same experience pool as imitation."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.experiment import provenance,measurement_lease
from rlmlir.predictor import CostPredictor,features,fit_predictor
from rlmlir.experience import load_pool, pool_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--pool-manifest",type=Path,help="pin the same initial experience as imitation")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    meta = provenance(ROOT)
    pool,sources = load_pool(args.runs,meta,manifest_path=args.pool_manifest)
    manifest = pool_manifest(pool,sources,meta)
    unique = {}
    for item in pool:
        key = (item["workload"]["id"],json.dumps(item["actions"],sort_keys=True))
        sample = unique.setdefault(key,{"workload_id":item["workload"]["id"],"split":item["workload"]["split"],
                                        "features":features(item["transitions"][-1]["next_observation"]),"samples":[]})
        sample["samples"].append([item["compile_install_ms"],item["execution_ms"]])
    for sample in unique.values():
        measurements = sample.pop("samples")
        sample["log_costs"] = [math.log(statistics.median(v[i] for v in measurements)) for i in range(2)]
    training = [e for e in unique.values() if e["split"] == "train"]
    validation = [e for e in unique.values() if e["split"] == "validation"]
    model = CostPredictor()
    history = fit_predictor(model,training,validation,seed=args.seed)
    args.output.mkdir(parents=True)
    (args.output/"initial-pool-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    metadata = {**meta,"schema_version":1,"sources":sources,"seed":args.seed,
                "initial_pool_fingerprint":manifest["fingerprint"],
                "training_examples":len(training),"validation_examples":len(validation),"history":history,
                "role":"supervised cost prediction; no deployment superiority claim"}
    torch.save({"state_dict":model.state_dict(),"metadata":metadata},args.output/"predictor.pt")
    (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"training_examples":len(training),"validation_examples":len(validation),"epochs":len(history)}))


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
