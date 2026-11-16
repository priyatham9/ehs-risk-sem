"""Export only the isolated weak-measurement stress study; older audits stay frozen."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_bootstrap_weak_measurement import check


def read_json(path):
    return json.loads(path.read_text())


def export(check_only=False):
    folder = ROOT / 'results/bootstrap_weak_measurement'
    protocol = read_json(folder / 'protocol.json')
    for relative, expected in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    complete = all((folder / name).exists() for name in ('summary.csv', 'paired.json', 'accounting.json', 'verification.json'))
    verification = check() if complete else None
    sources = [folder / name for name in ('protocol.json', 'timing_pilot.json', 'prior_stream_reference.json')]
    sources += [ROOT / name for name in ('tools/export_weak_measurement_audit.py',
                                        'tools/check_bootstrap_weak_measurement.py')]
    if complete:
        sources += [folder / name for name in ('summary.csv', 'paired.json', 'accounting.json', 'verification.json', 'recomputed_verification.json')]
    payload = {
        'schema': 'sem-weak-measurement-audit-v1', 'simulation_only': True, 'complete': complete,
        'manifest': protocol,
        'rows': json.loads(pd.read_csv(folder / 'summary.csv').to_json(orient='records', double_precision=15)) if complete else [],
        'paired': read_json(folder / 'paired.json') if complete else None,
        'accounting': read_json(folder / 'accounting.json') if complete else None,
        'verification': verification,
        'study_verification': read_json(folder / 'verification.json') if complete else None,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'interpretation': [
            'One prospectively specified stress condition chosen from the original weakest-measurement cell, not from these outcomes.',
            'All observations are generated; this is not evidence from a real workforce.',
            'All-attempt rejection includes skipped or invalid tests as nonrejections. Computable-only rejection has a different, selected denominator.',
            'Bootstrap refits are attempted only when the outer estimate and naive standard error are finite and the standard error is positive.',
            'Finite improper and nonconverged fits remain in the unchanged procedure. Their flags may overlap; computational completion is not calibration.',
            'Inner counts are nested within surveys and are descriptive, not independent binomial trials.',
            'Pointwise Monte Carlo intervals containing 5% do not establish equivalence or certify calibration. No alternative power is estimated.',
            'This experiment is separate from loading .75 studies; no pooling or paired comparison across experiments. Both loading and sample size differ, so cross-study differences do not isolate a loading effect.'
        ],
    }
    destination = ROOT / 'docs/weak-measurement-audit-data.json'
    content = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    if check_only:
        assert destination.read_text() == content, 'Public stress artifact is stale'
    else:
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(content)
        temporary.replace(destination)
    downloads = ROOT / 'docs/weak-measurement-downloads'
    downloads.mkdir(exist_ok=True)
    names = ['protocol.json', 'timing_pilot.json']
    if complete:
        names += ['summary.csv', 'paired.json', 'accounting.json', 'verification.json', 'recomputed_verification.json']
    for name in names:
        if check_only:
            assert (folder / name).read_bytes() == (downloads / name).read_bytes(), name
        else:
            shutil.copyfile(folder / name, downloads / name)
    print(json.dumps({'complete': complete, 'rows': len(payload['rows']), 'path': str(destination)}))
    return payload


if __name__ == '__main__':
    export(check_only='--check' in sys.argv)
