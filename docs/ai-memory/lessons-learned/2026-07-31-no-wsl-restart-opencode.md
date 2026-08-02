# No WSL restart while opencode is running — 2026-07-31

Date: 2026-07-31
Status: current
Scope: WSL workflow, opencode session management
Evidence: Removed WSL restart rules from foundation.md and operations/workflow.md

## The finding

The previous operational rule required restarting WSL (`wsl --shutdown`) before
every training, eval, or Python probe to avoid stale GPU/runtime state that could
cause TensorFlow to hang in `Dl+` (uninterruptible disk sleep).

This rule is no longer compatible with the current workflow because opencode now
runs inside WSL. Restarting WSL kills the opencode session, which terminates
the current task.

## Rules

1. **Do not restart WSL while opencode is running.** The session will be killed
   and any in-progress work will be lost.

2. **If stale WSL state causes TensorFlow hangs, diagnose first before
   restarting.** Check `nvidia-smi`, `VMmem` CPU usage, and process state
   before resorting to a WSL restart. Most hangs can be resolved by killing
   the stuck process rather than restarting the entire WSL instance.

3. **WSL restart is a last-resort recovery action.** Only use it when
   opencode is not running and there is no other way to recover from a stuck
   GPU or runtime state.

## What changed

- Removed "Restart WSL before every script launch" from `foundation.md`
- Removed "After every script that runs in WSL, shut WSL down again" from `foundation.md`
- Removed "For GPU-backed WSL retrains, always restart WSL" from `foundation.md`
- Removed "Operational lesson reinforced" section from `foundation.md`
- Removed WSL restart rules from `operations/workflow.md`
- This note preserves the historical context for future reference
