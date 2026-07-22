# Live contracts

Date: 2026-07-22  
Status: current  
Scope: STM32N657 camera and tip-focus deployment path

## Deployment contract

- The active tip-focus package is `tip_focus_v18_int8_n6_npu`.
- Its input is `224x224` color data.
- Its outputs are `56x56` center and tip heatmaps plus scalar `confidence` and
  `is_main_needle` values.
- Keep `c_info.json` and `network.csv` beside the raw xSPI2 blob in the Windows
  firmware package directory.
- The live path is OBB localizer followed by tip-focus. Keep the OBB-to-UNet
  handoff enabled; if OBB crop decoding fails, use the fixed training crop.

## Workspace contract

- Use WSL for ML training, export, and parity checks.
- Use the Windows checkout for firmware, CubeIDE, packaging inputs, and flashing.
- Treat the Windows firmware tree as the source of truth for board-flash inputs
  after a model is packaged.

## Evidence and change policy

This note is a compact pointer, not a substitute for generated metadata, build
logs, or board captures. When a contract changes, add a dated note under
`model-updates/` or `troubleshooting/`, then update this file only after the
new contract is verified.
