"""Run three small local extension checks sequentially, stopping on any error."""

import subprocess
import sys
from pathlib import Path

import yaml

from training import validate_config


CONFIGS = ["fc_mixed_smoke.yaml", "cae_mixed_smoke.yaml", "cae_conditioned_smoke.yaml"]


def main():
    root = Path(__file__).resolve().parents[1]
    # Validate every job before starting the first one.
    for name in CONFIGS:
        config = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
        validate_config(config)
        if config["purpose"] != "smoke":
            raise ValueError("This helper accepts smoke configurations only")
    for index, name in enumerate(CONFIGS, start=1):
        print(f"\nExtension check {index}/{len(CONFIGS)}: {name}", flush=True)
        result = subprocess.run([sys.executable, str(root / "src" / "train.py"),
                                 "--config", str(root / "configs" / name)], cwd=root)
        if result.returncode:
            print(f"Stopped at {name}; exit code {result.returncode}.", flush=True)
            return result.returncode
    print("\nEXTENSION CHECKS COMPLETED", flush=True)
    print("All three runs are debug-only. Keep their logs, checkpoints and figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
