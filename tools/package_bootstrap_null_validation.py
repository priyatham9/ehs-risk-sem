"""Package the completed fresh validation separately from the original study."""
from pathlib import Path
import hashlib
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_bootstrap_null_validation import check


def package():
    check()
    paths = list((ROOT / "ehs_risk_sem").glob("*.py"))
    paths += [ROOT / "simulations" / name for name in (
        "__init__.py", "dgp.py", "inference_distributions.py", "study_05_inference_audit.py",
        "study_06_bootstrap_ranking.py", "study_08_bootstrap_null_validation.py")]
    paths += [ROOT / "tests" / name for name in (
        "test_bootstrap_null_validation.py", "test_inference_distributions.py", "test_bootstrap_ranking.py", "test_inference_audit.py")]
    paths += [ROOT / "tools/check_bootstrap_null_validation.py", ROOT / "tools/package_bootstrap_null_validation.py"]
    paths += [p for p in (ROOT / "results/bootstrap_null_validation").iterdir() if p.is_file()]
    paths += [ROOT / "results/inference_audit/manifest.json", ROOT / "results/bootstrap_ranking/manifest.json", ROOT / "LICENSE"]
    destination = ROOT / "docs/model-audit-downloads/bootstrap-null-validation-replication.zip"
    destination.parent.mkdir(exist_ok=True)
    note = """Independent bootstrap null validation — separate follow-up
=========================================================
All observations are generated. No real workforce records are included.

This bundle contains the fixed 1,000-survey fresh-seed negative-null validation,
999 inner resamples per survey, and the paired first 399/all 999 comparison.
It is separate from the original 300-survey bootstrap; no pooling is performed.
The protocol was recorded before this follow-up's main computation, in response
to the original calibration concern. It is not a claim of historical original-
study preregistration.

Requirements: Python >= 3.9, numpy, pandas. Reference runtime Python 3.12.14,
numpy 2.3.5, pandas 2.2.3. No SciPy runtime dependency.

Verify without refitting models:
    python tools/check_bootstrap_null_validation.py
    python -m unittest discover -s tests -v

The verifier recomputes every stored 399/999 interval from its contrast draws,
checks source/dependency hashes and reconstructs all original-study seed IDs
from the original manifests to check fresh-seed independence.

To resume/reproduce in the reference runtime:
    OPENBLAS_NUM_THREADS=1 python -m simulations.study_08_bootstrap_null_validation --run --workers 4
To recompute from scratch or use a different runtime, first record a protocol in
a fresh directory with --out PATH --prepare, then use that --out PATH --run.
The fixed target is 1,000 surveys; no outcome-dependent early stopping.

The full original-study dataset is in the separate model-audit-replication.zip.
Its manifests are included here solely to verify seed identities. Original
findings are not overwritten by this validation. A rate interval containing 5%
is not an equivalence or calibration-certification result.
"""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        entries = [("REPLICATION.txt", note.encode())]
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
