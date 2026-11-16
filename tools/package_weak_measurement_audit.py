"""Package this stress study separately without duplicating older raw datasets."""
from pathlib import Path
import hashlib
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_bootstrap_weak_measurement import check


def package():
    check()
    paths = list((ROOT / 'ehs_risk_sem').glob('*.py'))
    paths += [ROOT / 'simulations' / name for name in (
        '__init__.py', 'dgp.py', 'inference_distributions.py', 'study_05_inference_audit.py',
        'study_06_bootstrap_ranking.py', 'study_08_bootstrap_null_validation.py',
        'study_09_bootstrap_weak_measurement.py')]
    paths += [ROOT / 'tests' / name for name in (
        'test_inference_audit.py', 'test_inference_distributions.py', 'test_bootstrap_ranking.py',
        'test_bootstrap_null_validation.py', 'test_bootstrap_weak_measurement.py',
        'test_weak_measurement_export.py')]
    paths += [ROOT / 'tools' / name for name in (
        'check_bootstrap_weak_measurement.py', 'package_weak_measurement_audit.py',
        'export_weak_measurement_audit.py', 'build_weak_measurement_note.py')]
    paths += [p for p in (ROOT / 'results/bootstrap_weak_measurement').iterdir() if p.is_file()]
    paths += [ROOT / 'LICENSE', ROOT / 'docs/weak-measurement-methods.md']
    note = """Weak-measurement bootstrap stress study — separate replication bundle
===================================================================
All observations are generated. This is one fixed positive equality null at
loading .50, N385, with 500 fresh outer surveys and 399 row resamples per survey.
Older studies are not pooled with these results.

Requirements: Python >= 3.9, NumPy and Pandas. Reference versions appear in the
prospective protocol. No SciPy runtime dependency.

This bundle independently verifies the saved-draw calculations:
    OPENBLAS_NUM_THREADS=1 python tools/check_bootstrap_weak_measurement.py
    OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v

The verifier recomputes every interval and decision, all conditional and
all-attempt rates, paired tables and Monte Carlo intervals, fit accounting,
source hashes and disjointness from the packaged prior stream-ID reference.
Original checkpoint hashes in that reference permit comparison against the
full repository. The bundle does not duplicate the older raw datasets.

For full model replay or exact checkpoint resume, use the repository checkout:
    https://github.com/priyatham9/ehs-risk-sem
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m simulations.study_09_bootstrap_weak_measurement --run --workers 4
The repository includes the older raw datasets and recorded pilots needed by
the frozen runner's prior-stream accounting. Do not interpret their absence
from this standalone verification bundle as a successful replay of that check.

For a new independent output directory, from a full checkout:
    python -m simulations.study_09_bootstrap_weak_measurement --out PATH --pilot
    python -m simulations.study_09_bootstrap_weak_measurement --out PATH --prepare
    python -m simulations.study_09_bootstrap_weak_measurement --out PATH --run --workers 4
Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for those commands. The pilot
retains timing only. Its prospective 500/250/no-run rule fixes sample size before
outcome inspection. The original recorded experiment remains fixed at 500.

Read docs/weak-measurement-methods.md for the estimator, failure gate, retained
improper/nonconverged fits, exact scope and interpretation limits. A rate
interval containing 5% is not equivalence or calibration certification.
"""
    destination = ROOT / 'docs/weak-measurement-downloads/replication.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        entries = [('REPLICATION.txt', note.encode())]
        entries += [(str(path.relative_to(ROOT)), path.read_bytes()) for path in sorted(set(paths))]
        for name, content in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 16, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content, compresslevel=9)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f'{digest}  {destination.name}\n')
    print(destination, destination.stat().st_size, digest)
    return destination


if __name__ == '__main__':
    package()
