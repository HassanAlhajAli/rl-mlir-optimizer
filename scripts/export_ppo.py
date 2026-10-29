"""Export a measured PPO checkpoint and prove whether optimizer updates occurred."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from rlmlir.policies import save_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    latest = MaskablePPO.load(args.run/"latest-policy.zip",device="cpu")
    initial = MaskablePPO.load(args.run/"initial-policy.zip",device="cpu")
    metadata = json.loads((args.run/"run.json").read_text())
    metadata.setdefault("observation_version",1)
    changed = [name for name,value in latest.policy.state_dict().items()
               if not torch.equal(value,initial.policy.state_dict()[name])]
    metadata.update(critic_fitted=False, critic_target="reference-relative PPO return, not adaptation log cost",
                    optimizer_epochs=latest._n_updates,changed_parameter_tensors=changed,
                    sb3_checkpoint_sha256=hashlib.sha256((args.run/"latest-policy.zip").read_bytes()).hexdigest())
    checksum = save_policy(latest.policy,args.output,metadata)
    args.output.with_suffix(".json").write_text(json.dumps({**metadata,"policy_sha256":checksum},indent=2)+"\n")
    print(json.dumps({"optimizer_epochs":latest._n_updates,"changed_parameter_tensors":len(changed),"policy_sha256":checksum}))


if __name__ == "__main__":
    main()
