# Development split protection and final evaluation status

Held-out performance remains sealed. The full method freeze and final measurement
orchestrator are not yet implemented, and the interrupted training study is not
ready for final evaluation. These safeguards strengthen the existing development
boundary; they do not declare a completed evaluation gate.

## Workload identity comes from the frozen manifest

`FrozenCorpus` checks the actual corpus bytes against their saved SHA256 and
rejects duplicate IDs, duplicate family/shape pairs and related groups crossing
splits. A workload's ID, family, dimensions, group and split must all match the
manifest. Returned workload objects cannot modify the stored identity.

`Evaluator.measure()` checks this identity before consuming a request, using a
cached outcome or contacting the compiler. Changing a held-out entry's label to
`train` does not grant access. New measurement records include the corpus hash.
An imported cache must contain correct outcomes for the same compiler, canonical
development workloads and matching source/action measurement identities. Older
records without the per-record corpus hash still require exact manifest identity
and the source/action key; their original artifacts are not rewritten.

Imitation and cost-predictor experience loading also checks the authoritative
workload identity and joins each trajectory to a correct measurement of that
same workload, action sequence and compiler. A successful row in a different
program's measurement table cannot be substituted for the trajectory's outcome.

PPO continuation and validation selection use the same full compatibility check
as adaptation: compiler binary, corpus, feature version, measurement protocol,
compiler release and target. Provenance hashes the verified corpus contents,
rather than merely trusting the sidecar's text.

These are safeguards against programming and data-handling mistakes, not a
security boundary against somebody rewriting the code or manifest. Correctness
tests may inspect held-out shape metadata but do not time those programs.
Standalone compiler correctness tools still accept arbitrary small test kernels.

## What is still required before opening final performance

- Resolve the current compiler continuation gate and freeze the study allocation.
- Complete and account for both PPO objectives and all five seeds, with checkpoint
  selection on validation separately for each run.
- Freeze the fixed, heuristic, predictor, PPO and imitation selectors, and the
  equal-budget adaptation/search controls with compatible model/data identities.
- Save the exact method configurations, artifact hashes, seed lists, reuse
  contexts, budget rules, aggregation and uncertainty procedure before evaluation.
- Implement and check five-round interleaving, fresh complete deployments,
  reference drift handling, failures, cost accounting and bounded resumption.

There is deliberately no `allow_heldout` switch in the development evaluator.
A future final evaluator needs its own reviewed frozen-study contract. Until
that implementation and its prerequisites are complete, this document grants
no held-out measurement access.

## Checks

`tests/test_corpus.py` uses rejected inputs and synthetic experience records to
exercise relabeling, changed dimensions, altered hashes, ambiguous identities,
incorrect cached outcomes and wrong trajectory/measurement joins. A compiler
stub raises if any invalid case gets as far as a compiler request.

```sh
.venv/bin/python -m pytest tests/test_learning.py tests/test_adaptation.py \
  tests/test_predictor.py tests/test_queue.py tests/test_corpus.py -q
```

The saved integration and interrupted PPO auditors apply the same manifest
guard without executing generated code. Their evidence is retained separately
from the original experiment records.
