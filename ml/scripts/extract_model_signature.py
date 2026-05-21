"""Extract xSPI2 model blob signatures for firmware constant updates.

Usage:
    python extract_model_signature.py <path_to_raw_file>

Prints the first 16 and last 16 bytes of the .raw blob in C hex-literal format,
suitable for pasting into app_ai.c signature arrays.
"""

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python extract_model_signature.py <path_to_raw_file>")
        sys.exit(1)

    raw_path = Path(sys.argv[1])
    if not raw_path.is_file():
        print(f"ERROR: File not found: {raw_path}")
        sys.exit(1)

    data = raw_path.read_bytes()
    head = data[:16]
    tail = data[-16:]

    print(f"File: {raw_path}")
    print(f"Size: {len(data)} bytes")
    print()
    print("Head 16 bytes (C format):")
    head_hex = ", ".join(f"0x{b:02X}U" for b in head)
    print(f"    {head_hex}")
    print()
    print("Tail 16 bytes (C format):")
    tail_hex = ", ".join(f"0x{b:02X}U" for b in tail)
    print(f"    {tail_hex}")
    print()
    print("Update these arrays in app_ai.c:")
    print("  app_ai_xspi2_signature_start[]")
    print("  app_ai_xspi2_signature_tail[]")


if __name__ == "__main__":
    main()
