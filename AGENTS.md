# Project working rules

Read `docs/status.md` and `docs/protocol.md` before implementation or experiments. The user's explicit instructions take precedence.

- Explain meaningful changes to Hassan: purpose, important files, correctness evidence, limitations. Keep personal learning and interview notes in the sibling `../private-notes/` directory, never this repository.
- Keep compiler transformations, executable ownership, timing, learning and reporting separate. Use C++ RAII and structured errors. Comment non-obvious invariants and measurement boundaries. Prefer a small readable implementation over speculative abstractions.
- Use the pinned toolchain. Do not silently change floating-point semantics, targets, action legality or evaluation budgets to obtain a win.
- Verify transformed IR and independent numerical correctness before accepting timing. Do not loosen tolerances to pass a candidate. Preserve failed attempts and diagnose repeated failures.
- Treat held-out performance as sealed until all methods are frozen. Never put identifiers, split labels, reference best actions or held-out timings into observations.
- Count compilation, selection, trials, updates, cache hits and deployment according to the protocol. Never present extra search as evidence of better learning without an equal-budget control.
- Benchmark serially without training/build contention. Save raw observations, hashes, versions, seeds and errors. CI correctness tests do not establish performance.
- Make coherent reviewable commits, run relevant checks, inspect the diff and preserve corrections. State exactly which milestones are implemented and which are only proposed.
- Do not claim novelty, RL superiority or Arm performance without supporting evidence. Negative results are publishable.
