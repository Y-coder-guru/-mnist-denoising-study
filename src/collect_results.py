"""Archive formal outputs for backup after each Kaggle batch."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/formal"))
    args = parser.parse_args()
    root = args.root.resolve()
    work = (args.work_dir if args.work_dir.is_absolute() else root / args.work_dir).resolve()
    if not work.is_dir():
        raise FileNotFoundError(f"No formal outputs: {work}")
    target = root / "dist"
    target.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = target / f"formal_results_{timestamp}.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                if not path.resolve().is_relative_to(work):
                    raise ValueError("Result file resolves outside the output directory")
                archive.write(path, "formal/" + path.relative_to(work).as_posix())
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Result ZIP failed CRC verification")
    print(f"RESULT BACKUP READY: {destination}")
    print("Archive at an epoch boundary or after the queue stops for consistent checkpoint/log exports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
