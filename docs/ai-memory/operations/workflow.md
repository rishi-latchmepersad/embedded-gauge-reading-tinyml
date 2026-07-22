# Workflow and WSL Notes

This file holds the operational workflow and WSL-specific rules.
See `archive.md` for the full chronology.

## Current Build / Tooling

- Use `poetry` for env management and scripts.
- Prefer `pytest` for tests.
- Use WSL for ML work, with the GPU preferred.
- Always run jobs in bash scripts inside WSL, and tail the logs so you can see when they hang or fail.

## WSL Invocation Pattern

- Always prefer the Ubuntu WSL distro explicitly.
- Bare `wsl bash ...` can land in the wrong distro if `docker-desktop` is the default.
- PowerShell launchers should call `wsl.exe -d Ubuntu-24.04 --exec /bin/bash -lc ...` when they need to run a script.

## WSL Reliability Rules

- Restart WSL before every script launch.
- Do not reuse a prior WSL session between script runs because stale state can hang TensorFlow or GPU startup.
- After every script that runs in WSL, shut WSL down again with `wsl --shutdown`.
- For long-running WSL jobs, treat the Windows `VMmem` process as the liveness signal.
- If `VMmem` CPU stays below 2%, assume the command is effectively stuck or idle and stop waiting on it.
- TensorFlow GPU startup in WSL can stall inside the runtime GPU probe.
- For GPU-backed retrains, a fresh WSL session is part of the reliable recipe.

## Launcher Notes

- The repo now has PowerShell launchers for the long-term training runs that force Ubuntu explicitly.
- Keep launcher scripts boring and explicit; do not depend on ambient distro defaults.

## Repackaging Workflow

- Prefer a WSL script for board packaging and export work.
- Use a Windows-writable staging path for pack steps that need to emit files outside Linux paths.
- The relocatable packaging flow needs a staging build directory that the pack tool can write to cleanly.

## Logging Convention

- Important runs should leave a logfile under `ml/artifacts/training_logs/`.
- Keep the wrapper responsible for teeing stdout/stderr so hang diagnosis is easy later.
- If a script is likely to stall, emit a short wrapper banner before the long-running command starts.
