"""Extract xSPI2 model blob signatures for firmware constant updates.

This is a small Windows-friendly helper that prints the first and last 16 bytes
of a model blob so the flash script can verify the deployed model signatures
without depending on the ML checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Print the head and tail bytes of a model blob."""
    if len(sys.argv) < 2:
        print("Usage: python extract_model_signature.py <path_to_raw_file>")
        return 1

    raw_path = Path(sys.argv[1])
    if not raw_path.is_file():
        print(f"ERROR: File not found: {raw_path}")
        return 1

    data = raw_path.read_bytes()
    head = data[:16]
    tail = data[-16:]

    print(f"File: {raw_path}")
    print(f"Size: {len(data)} bytes")
    print()
    print("Head 16 bytes (C format):")
    print("    " + ", ".join(f"0x{byte:02X}U" for byte in head))
    print()
    print("Tail 16 bytes (C format):")
    print("    " + ", ".join(f"0x{byte:02X}U" for byte in tail))
    print()
    print("Update these arrays in app_ai.c:")
    print("  app_ai_xspi2_signature_start[]")
    print("  app_ai_xspi2_signature_tail[]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
