# ML Scripts

## Current entry points

- `run_wsl_guarded.sh` is the required process, memory, and concurrency guard.
- `wsl_ml.sh` is a compatibility command router; its workload branches
  re-enter through `run_wsl_guarded.sh`.
- New training, evaluation, conversion, and packaging workflows should be
  added as one explicit command under the guard, not as a new ambient shell
  loop.

## Historical wrappers

The many `run_*.sh`, `auto_*.sh`, and `test_*.py` files preserve prior model
families and board experiments. Some refer to old `/mnt/d` paths, old scalar
artifacts, or intentionally parallel launches that predate the memory guard.
They are not current pipeline contracts. Run them only after checking their
inputs and by placing the complete invocation under `run_wsl_guarded.sh`.

The active candidate and its exact commands are documented in
`docs/ai-memory/current-state/ml-pipeline.md`.
