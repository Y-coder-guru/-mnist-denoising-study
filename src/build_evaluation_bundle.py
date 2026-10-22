"""Package audited training archives plus code/data, without running model inference."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_bundle import bundle_members
from data_pipeline import load_saved_split, verify_raw_integrity
from experiments import canonical_hash, file_hash, make_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / "outputs/review/seed3407/audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["formal_runs"], audit["pilot_runs"]) != (57, 3):
        raise ValueError("The full training audit must be complete first")
    # Verify every archive and compare extracted files again before selecting input records.
    for record in audit["archives"]:
        path = root / "dist" / record["archive"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Audited archive changed: {path}")
        with ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/") and archive.read(name) != (path.with_suffix("") / name).read_bytes():
                    raise ValueError(f"Extracted archive differs: {name}")
    load_saved_split(root)
    verify_raw_integrity(root)  # byte hashes only, no model evaluation of the test set
    members = {p.relative_to(root).as_posix(): p for p in bundle_members(root)}
    inventories = audit["inventory"]
    if len(inventories) != 60 or len({r["job_id"] for r in inventories}) != 60:
        raise ValueError("Training inventory is not the expected 60 unique runs")
    for row in inventories:
        run = Path(row["run"]).resolve()
        if not run.is_relative_to(root / "dist") or file_hash(run / "last.pt") != row["last_checkpoint_sha256"]:
            raise ValueError(f"Audited run has changed: {row['job_id']}")
        for path in run.rglob("*"):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(run):
                    raise ValueError("Unexpected link in a training run")
                name = f"outputs/formal/runs/{row['job_id']}/" + path.relative_to(run).as_posix()
                members[name] = path
    old = root / "dist/formal_results_20261021T134841789597Z/formal"
    members["outputs/formal/gradient_selection.json"] = old / "gradient_selection.json"
    members["docs/training_audit.json"] = audit_path
    generated = {"outputs/formal/task_matrix.json": (json.dumps(make_jobs(), indent=2) + "\n").encode()}
    hashes = {name: file_hash(path) for name, path in sorted(members.items())}
    hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in generated.items()})
    environment = json.loads((old / "runs/fc_baseline_s03_z16_seed42/environment.json").read_text(encoding="utf-8"))
    version = canonical_hash(hashes)[:12]
    manifest = {"kind": "mnist_final_evaluation", "version": version,
                "formal_runs": 57, "pilot_runs": 3, "files_sha256": hashes,
                "training_environment": environment, "source_archives": audit["archives"],
                "note": "Complete original training artifacts; no final test scores yet. Run src/run_final_evaluation.py on Kaggle GPU. No training is started."}
    destination = root / "dist" / f"mnist_final_eval_{version}.zip"
    temporary = destination.with_suffix(".zip.tmp")
    print(f"Writing {len(hashes)} files into {destination.name}", flush=True)
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=4) as archive:
        for name, path in sorted(members.items()):
            archive.write(path, "mnist_final_eval/" + name)
        for name, data in generated.items():
            archive.writestr("mnist_final_eval/" + name, data)
        archive.writestr("mnist_final_eval/final_evaluation_manifest.json", json.dumps(manifest, indent=2).encode())
    print("Verifying all ZIP members...", flush=True)
    with ZipFile(temporary) as archive:
        if len(archive.namelist()) != len(hashes) + 1:
            raise ValueError("Unexpected archive member count")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read("mnist_final_eval/" + name)).hexdigest() != expected:
                raise ValueError(f"ZIP member failed verification: {name}")
    temporary.replace(destination)
    receipt = {"zip": str(destination), "sha256": file_hash(destination),
               "bytes": destination.stat().st_size, "version": version,
               "manifest_sha256": hashlib.sha256(json.dumps(manifest, indent=2).encode()).hexdigest(),
               "formal_runs": 57, "pilot_runs": 3, "member_count": len(hashes) + 1}
    (root / "outputs/review/seed3407/final_evaluation_bundle.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print("FINAL TEST BUNDLE READY. No training, data download, or test inference was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
