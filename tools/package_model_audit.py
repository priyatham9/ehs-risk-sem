"""Create a deterministic, self-contained audit replication bundle after completion."""
from pathlib import Path
import hashlib
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_inference_audit import check


def package():
    check(ROOT / "results/inference_audit", 54, 1000)
    check(ROOT / "results/bootstrap_ranking", 4, 300, bootstrap=True)
    paths = list((ROOT / "ehs_risk_sem").glob("*.py"))
    paths += [ROOT / "simulations" / name for name in (
        "__init__.py", "dgp.py", "inference_distributions.py", "study_05_inference_audit.py",
        "study_06_bootstrap_ranking.py", "study_07_information_bound.py", "validate_information_bound.py",
        "study_08_bootstrap_null_validation.py")]
    paths += [ROOT / "tests" / name for name in (
        "test_inference_audit.py", "test_bootstrap_ranking.py", "test_information_bound.py", "test_inference_distributions.py",
        "test_bootstrap_null_validation.py")]
    paths += [ROOT / "tools" / name for name in (
        "check_inference_audit.py", "convergence_sensitivity.py", "plot_inference_audit.py",
        "export_model_audit.py", "build_model_audit_note.py", "package_model_audit.py")]
    for folder in ("inference_audit", "bootstrap_ranking", "information_bound"):
        paths += [p for p in (ROOT / "results" / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [ROOT / "docs/model-audit-methods.md", ROOT / "LICENSE",
              ROOT / "results/bootstrap_null_validation/protocol.json"]
    destination = ROOT / "docs/model-audit-downloads/model-audit-replication.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    readme = """SEM coefficient-contrast audit replication bundle
================================================
All records are simulated. No workplace records are included.

Requires Python>=3.9, numpy and pandas. Optional figures require matplotlib.
Reference runtime: Python3.12.14, numpy2.3.5, pandas2.2.3. Checkpoint manifests
guard runtime versions; use a fresh --out directory if the runtime differs.
See docs/model-audit-methods.md for design, inferential limits and provenance.

From the extracted root, verify saved artifacts without rerunning simulations:
    python tools/check_inference_audit.py
    python -m unittest discover -s tests -v

Rerun/resume the exact checkpointed studies:
    OPENBLAS_NUM_THREADS=1 python -m simulations.study_05_inference_audit --reps 1000 --workers 2
    OPENBLAS_NUM_THREADS=1 python -m simulations.study_06_bootstrap_ranking --reps 300 --n-boot 399 --workers 4
    OPENBLAS_NUM_THREADS=1 python -m simulations.study_07_information_bound
    OPENBLAS_NUM_THREADS=1 python -m simulations.validate_information_bound

Use a fresh --out directory for a new seed or modified design. Existing manifests
prevent accidentally mixing incompatible simulations. To recompute every draw
from scratch under the same design, choose a fresh output directory.

Original execution manifests/source snapshots and distribution_migration.json
preserve the documented switch from SciPy reference distributions to numerically
equivalent standard-library functions. No saved decisions changed. The full
measurement and structural estimator is unchanged. Bootstrap resamples skip
unused global-fit diagnostics only; paired tests check identical estimates.

The separate fresh-seed null-validation protocol and source are included, but
this original-study bundle contains no partial validation draws. Its completed
results will be offered separately.

Public data/figures can be rebuilt using tools/export_model_audit.py and
(optional matplotlib) tools/plot_inference_audit.py. Methods note links refer to
the public workbench; the same CSVs are under results/ in this bundle.
"""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        entries = [("REPLICATION.txt", readme.encode())]
        entries += [(str(path.relative_to(ROOT)), path.read_bytes()) for path in sorted(set(paths))]
        for name, content in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 16, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(f"{digest}  {destination.name}\n")
    print(destination, destination.stat().st_size, digest)
    return destination


if __name__ == "__main__":
    package()
