"""Fit the non-RL cost baseline on the same experience pool as imitation."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.experiment import provenance,measurement_lease
from rlmlir.predictor import CostPredictor,cost_examples,fit_predictor
from rlmlir.experience import load_pool, pool_manifest
from rlmlir.rechecks import load_rechecks, measurement_access


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--pool-manifest",type=Path,help="pin the same initial experience as imitation")
    parser.add_argument("--rechecks",type=Path,help="complete shared recheck directory from imitation")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    meta = provenance(ROOT)
    pool,sources = load_pool(args.runs,meta,manifest_path=args.pool_manifest)
    manifest = pool_manifest(pool,sources,meta)
    recheck_manifest = None
    if args.rechecks is not None:
        rechecked,recheck_manifest = load_rechecks(args.rechecks,pool,manifest,meta)
        pool = pool + rechecked
    examples = cost_examples(pool)
    training = [e for e in examples if e["split"] == "train"]
    validation = [e for e in examples if e["split"] == "validation"]
    model = CostPredictor()
    history = fit_predictor(model,training,validation,seed=args.seed)
    args.output.mkdir(parents=True)
    (args.output/"initial-pool-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    metadata = {**meta,"schema_version":1,"sources":sources,"seed":args.seed,
                "initial_pool_fingerprint":manifest["fingerprint"],
                "measurement_access":measurement_access(manifest,recheck_manifest),
                "target_aggregation":"median per program/history over distinct source-measurement identities",
                "training_examples":len(training),"validation_examples":len(validation),"history":history,
                "role":"supervised cost prediction; no deployment superiority claim"}
    torch.save({"state_dict":model.state_dict(),"metadata":metadata},args.output/"predictor.pt")
    if recheck_manifest is not None:
        (args.output/"shared-rechecks.json").write_text(json.dumps(recheck_manifest,indent=2)+"\n")
    (args.output/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"training_examples":len(training),"validation_examples":len(validation),"epochs":len(history)}))


if __name__ == "__main__":
    with measurement_lease(ROOT):
        main()
