# ML Tests

`tests/smoke/` is the default pytest collection. It must remain fast,
deterministic, and runnable from a fresh checkout without ignored model
artifacts.

The other tests in this directory were written for older experiment contracts
or require local checkpoints/manifests. They remain useful when reviving that
experiment, but are not a claim that every historical model family is current.
Run them explicitly by path and record any required artifact in the relevant
model-update note.
