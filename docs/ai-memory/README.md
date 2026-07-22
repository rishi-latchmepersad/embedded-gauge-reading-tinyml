# Project AI Memory

This directory is the working memory for AI agents and teammates. Keep the
root index short; put durable knowledge in the most specific topic folder.

## Where to write

| Folder | Use it for |
| --- | --- |
| [`current-state/`](current-state/) | Facts describing the currently supported pipeline, hardware, data, or contracts |
| [`model-updates/`](model-updates/) | Training runs, metrics, artifacts, model decisions, and deployment candidates |
| [`troubleshooting/`](troubleshooting/) | Symptoms, root causes, diagnostics, and reusable fixes |
| [`lessons-learned/`](lessons-learned/) | General rules that should influence future work |
| [`operations/`](operations/) | Repeatable workflows, handoffs, commands, and verification checklists |
| [`archive/`](archive/) | Superseded chronology and legacy snapshots |

## Entry format

Every new note should include:

- `Date:` in `YYYY-MM-DD` format.
- `Status:` one of `current`, `validated`, `experimental`, or `superseded`.
- `Scope:` the model, board, dataset, or workflow affected.
- `Evidence:` links to source files, logs, metrics, commits, or captures.
- `Decision:` the practical action future work should take.

Use one note per decision or incident. Prefer descriptive filenames such as
`2026-07-22-obb-reloc-init-order.md`; do not append routine updates to a
single ever-growing file.

## Reading order

1. Read `current-state/` for the live contracts.
2. Read relevant `troubleshooting/` and `lessons-learned/` notes before changing firmware or deployment behavior.
3. Read `model-updates/` for candidate history and metrics.
4. Use `archive/` only to recover older context.

The former monolithic memory file is preserved at
[`archive/legacy-ai-memory-2026-07-22.md`](archive/legacy-ai-memory-2026-07-22.md).
It is not the place for new notes.
