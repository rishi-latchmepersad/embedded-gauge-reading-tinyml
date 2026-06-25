"""Stage the board-bbox OBB deploy candidate into the STM32N6 firmware tree."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
SOURCE_DIR: Path = REPO_ROOT / "tmp" / "stedgeai_obb_box_board_bbox_deploy_candidate_out"
FIRMWARE_PACKAGE_DIR: Path = (
	REPO_ROOT
	/ "firmware"
	/ "stm32"
	/ "n657"
	/ "st_ai_output"
	/ "packages"
	/ "obb_box_board_bbox_deploy_candidate"
)
STAI_OUTPUT_DIR: Path = FIRMWARE_PACKAGE_DIR / "st_ai_output"
MODEL_NAME: str = "obb_box_board_bbox_deploy_candidate"
RAW_NAME: str = f"{MODEL_NAME}_atonbuf.xSPI2.raw"
PROBE_BYTES: int = 16


def _copy_file(src: Path, dst: Path) -> None:
	"""Copy one artifact and keep the destination tree deterministic."""
	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dst)
	print(f"[PACKAGE] {src.name} -> {dst}", flush=True)


def _print_signature(raw_path: Path) -> None:
	"""Print the raw blob signature so flash_boot.bat stays easy to update."""
	data = raw_path.read_bytes()
	sha16 = hashlib.sha256(data).hexdigest()[:16]
	head = data[:PROBE_BYTES]
	tail = data[-PROBE_BYTES:]
	print(f"[PACKAGE] sha256[:16] = {sha16}")
	print(f"[PACKAGE] size = {len(data)} bytes")
	print(f"[PACKAGE] start ({PROBE_BYTES} bytes):")
	print("    " + " ".join(f"0x{byte:02X}U," for byte in head))
	print(f"[PACKAGE] tail ({PROBE_BYTES} bytes):")
	print("    " + " ".join(f"0x{byte:02X}U," for byte in tail))


def main() -> None:
	"""Copy the generated ST Edge AI artifacts into the firmware package tree."""
	required_files = (
		SOURCE_DIR / f"{MODEL_NAME}.c",
		SOURCE_DIR / f"{MODEL_NAME}.h",
		SOURCE_DIR / RAW_NAME,
		SOURCE_DIR / "LICENSE.txt",
	)
	for path in required_files:
		if not path.is_file():
			raise FileNotFoundError(f"Required artifact not found: {path}")

	if FIRMWARE_PACKAGE_DIR.exists():
		shutil.rmtree(FIRMWARE_PACKAGE_DIR)
	STAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for name in (f"{MODEL_NAME}.c", f"{MODEL_NAME}.h", RAW_NAME, "LICENSE.txt"):
		_copy_file(SOURCE_DIR / name, STAI_OUTPUT_DIR / name)

	_print_signature(STAI_OUTPUT_DIR / RAW_NAME)
	print("[PACKAGE] Done.", flush=True)


if __name__ == "__main__":
	main()
