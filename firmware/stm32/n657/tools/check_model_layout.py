#!/usr/bin/env python3
"""Validate an stedgeai N6 package's NPU memory usage against the firmware's
reserved CPU/DMA regions before flashing a new model.

The lesson from 2026-08-05: every wrong-reading bug came from an NPU
activation pool overlapping a CPU/DMA buffer (the DMA frame, the keypoint
staging shadow, the snapshot). The models' WRITTEN addresses are the
0x34xxxxxx/0x24xxxxxx literals in the generated `*_reloc.c`; the pool
allocations in the same file (name=/offset=/size= comment lines) are
worst-case and larger than the real usage.

Run this against a NEW package before flashing it:
    python tools/check_model_layout.py ^
        st_ai_output/packages/ellipse_iter8_universal_wide_deep_int8_n6_npu

Keep the RESERVED list in sync with STM32N657X0HXQ_LRUN.ld and the section
attributes in app_ai.c / app_camera_buffers.c.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (name, start, end_exclusive) - the firmware's CPU/DMA-owned windows.
RESERVED: list[tuple[str, int, int]] = [
    ("app code (.text/.rodata)", 0x34000400, 0x34096400),
    ("app .data/.bss + main stack (RAM)", 0x34099400, 0x340FFC00),
    ("camera pad + DMA buffer (RAM_NC)", 0x24160000, 0x24200000),
    ("AXISRAM_NS scratch (unused snapshot etc.)", 0x24200000, 0x242A0000),
    ("keypoint input shadow (.npusram6)", 0x3439C000, 0x343A8400),
    ("npusram6 remainder", 0x343A8400, 0x343C0000),
    ("npupolar (NPU_SRAM6_POLAR)", 0x34350000, 0x34369000),
]

# Pool descriptor comment line: index=5 ... name=cpuRAM1 offset=0x34064000 ...
POOL_RE = re.compile(
    r"index=(\d+).*?name=(\S+).*?offset=0x([0-9a-fA-F]{8}).*?size=(\d+)"
)
# Tensor address literal: .addr_base = {(unsigned char *)(0x342e0000UL)}
TENSOR_RE = re.compile(r"0x(3[0-9a-fA-F]{7}|2[0-9a-fA-F]{7})UL")


def parse_reloc(reloc_c: Path) -> tuple[list[tuple[int, int]], list[int]]:
    """Return (pool allocations [(base, size)], tensor addresses)."""
    pools: list[tuple[int, int]] = []
    tensors: list[int] = []
    for line in reloc_c.read_text(encoding="utf-8", errors="replace").splitlines():
        pool_match = POOL_RE.search(line)
        if pool_match:
            size = int(pool_match.group(4))
            if size > 0:
                pools.append((int(pool_match.group(3), 16), size))
            continue
        tensor_match = TENSOR_RE.search(line)
        if tensor_match:
            tensors.append(int(tensor_match.group(1), 16))
    return sorted(pools), sorted(set(tensors))


def overlaps(addr: int, size: int, start: int, end: int) -> bool:
    """True when [addr, addr+size) intersects [start, end)."""
    return (addr < end) and (start < addr + size)


def main() -> None:
    """Run the layout gate for one or more packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packages",
        nargs="+",
        type=Path,
        help="stedgeai package directories (st_ai_output/packages/<name>).",
    )
    args = parser.parse_args()

    failures = 0
    for pkg in args.packages:
        pkg = pkg.resolve()
        print(f"== {pkg.name} ==")
        reloc_candidates = list(pkg.glob("st_ai_ws/build_network/*_reloc.c"))
        if not reloc_candidates:
            print("   ERROR: no *_reloc.c found; cannot check layout.")
            failures += 1
            continue
        pools, tensors = parse_reloc(reloc_candidates[0])
        print(f"   {len(tensors)} tensor addresses, {len(pools)} pools")

        pkg_fail = 0
        for addr in tensors:
            for name, start, end in RESERVED:
                if overlaps(addr, 0x10, start, end):
                    print(f"   FAIL: tensor base 0x{addr:08X} sits inside "
                          f"'{name}' (0x{start:08X}..0x{end:08X})")
                    pkg_fail += 1
        for base, size in pools:
            for name, start, end in RESERVED:
                if overlaps(base, size, start, end):
                    print(f"   WARN: pool 0x{base:08X} (+0x{size:X}) reaches "
                          f"'{name}' - verify the real tensor usage before "
                          "flashing (allocations are worst-case)")
        if pkg_fail == 0:
            print("   PASS: no tensor base overlaps a reserved CPU/DMA region")
        failures += pkg_fail

    print("")
    if failures:
        print(f"LAYOUT CHECK FAILED ({failures} overlaps). "
              "Do NOT flash: adjust the package pools or the firmware layout "
              "first (see docs/ai-memory.md prevention checklist).")
        sys.exit(1)
    print("LAYOUT CHECK PASSED. Verify on-target with the diag_ct_in*.bin "
          "parity comparison after flashing.")
    sys.exit(0)


if __name__ == "__main__":
    main()
