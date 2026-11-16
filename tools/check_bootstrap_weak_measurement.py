"""Recompute the completed weak-measurement stress audit from saved draws."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.study_05_inference_audit import Z975, seed_for
from simulations.study_09_bootstrap_weak_measurement import (
    CELL, STAGE, MAIN_SEED, INNER_REPS, PILOT_SEED, PILOT_REPS, PREVIOUS, summaries,
)


def make_seed_reference():
    """Keep prior stream identities available in a standalone follow-up bundle."""
    identifiers, sources = set(), {}
    for directory in PREVIOUS:
        for path in sorted((ROOT / directory).rglob('*.jsonl')):
            content = path.read_bytes()
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
            for line in content.decode().splitlines():
                row = json.loads(line)
                identifiers.update(row[key] for key in ('seed', 'bootstrap_seed') if key in row)
    assert sources
    for rep in range(PILOT_REPS):
        seed = seed_for(PILOT_SEED, CELL, STAGE, rep)
        identifiers.add(seed)
        identifiers.add(np.random.SeedSequence([seed, 606]).generate_state(1, dtype=np.uint64)[0].item())
    reference = {'previous_checkpoint_sha256': sources, 'stream_ids': sorted(identifiers),
                 'includes_new_timing_pilot': True,
                 'purpose': 'Exact prior stream identities, including recorded pilots, for standalone seed-disjointness checks; original checkpoint hashes permit verification against the prior replication bundles.'}
    path = ROOT / 'results/bootstrap_weak_measurement/prior_stream_reference.json'
    content = json.dumps(reference, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == content, 'Previous seed reference changed'
    else:
        path.write_text(content)
    return {'previous_stream_ids': len(identifiers), 'previous_checkpoint_files': len(sources)}


def check():
    folder = ROOT / 'results/bootstrap_weak_measurement'
    plan = json.loads((folder / 'protocol.json').read_text())
    assert plan['outer_reps'] == 500 and plan['inner_reps'] == INNER_REPS == 399
    assert plan['n'] == 385 and plan['loading'] == .5 and plan['pair'] == 'positive'
    assert plan['seed'] == MAIN_SEED and plan['stage'] == STAGE
    for path, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256((folder / 'timing_pilot.json').read_bytes()).hexdigest() == plan['timing_pilot_sha256']
    paths = sorted(folder.glob('chunk_*.jsonl'))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(paths) == 4 and len(rows) == 500
    assert sorted(row['rep'] for row in rows) == list(range(500))
    stream_ids = set()
    for row in rows:
        assert row['seed'] == seed_for(MAIN_SEED, CELL, STAGE, row['rep'])
        assert row['n'] == 385 and row['loading'] == .5 and row['true_difference'] == 0
        assert row['pair'] == 'positive' and row['stage'] == STAGE and row['n_boot'] == INNER_REPS
        assert row['cell'] == CELL.key and row['multiplier'] == CELL.multiplier
        assert row['seed'] not in stream_ids
        stream_ids.add(row['seed'])
        draws = np.array(row['bootstrap_draws'], dtype=float)
        assert np.isfinite(draws).all()
        assert 0 <= row['bootstrap_failed'] <= INNER_REPS and len(draws) <= INNER_REPS
        assert row['bootstrap_attempted'] == len(draws) + row['bootstrap_failed']
        if row['observed_completed']:
            expected_valid = bool(np.isfinite(row['observed_estimate']) and
                                  np.isfinite(row['observed_se']) and row['observed_se'] > 0)
            assert row['observed_valid'] == expected_valid
        else:
            assert not row['observed_valid']
        for field in ('bootstrap_improper', 'bootstrap_nonconverged'):
            assert 0 <= row[field] <= len(draws)
        if row['observed_valid']:
            assert np.isfinite(row['observed_estimate']) and row['observed_se'] > 0
            assert row['observed_reject'] == (abs(row['observed_estimate']/row['observed_se']) > Z975)
            seed = np.random.SeedSequence([row['seed'], 606]).generate_state(1, dtype=np.uint64)[0].item()
            assert seed == row['bootstrap_seed'] and seed not in stream_ids
            stream_ids.add(seed)
            assert row['bootstrap_attempted'] == INNER_REPS
        else:
            assert not row['observed_reject'] and not row['bootstrap_valid']
            assert row['bootstrap_attempted'] == 0
        computable = bool(len(draws) >= max(20, .95 * INNER_REPS))
        assert row['bootstrap_valid'] == computable
        if computable:
            lo, hi = np.quantile(draws, [.025, .975])
            se = draws.std(ddof=1)
            nlo, nhi = row['observed_estimate'] - Z975*se, row['observed_estimate'] + Z975*se
            np.testing.assert_allclose([row['percentile_low'], row['percentile_high'], row['bootstrap_se'], row['normal_low'], row['normal_high']], [lo, hi, se, nlo, nhi], rtol=1e-12, atol=1e-12)
            assert row['percentile_reject'] == bool(lo > 0 or hi < 0)
            assert row['normal_reject'] == bool(nlo > 0 or nhi < 0)
            assert row['percentile_coverage'] == bool(lo <= 0 <= hi)
            assert row['normal_coverage'] == bool(nlo <= 0 <= nhi)
        else:
            assert not row['percentile_reject'] and not row['normal_reject']
    reference_path = folder / 'prior_stream_reference.json'
    reference = json.loads(reference_path.read_text())
    assert not stream_ids.intersection(reference['stream_ids'])
    checked_previous_files = 0
    for relative, digest in reference['previous_checkpoint_sha256'].items():
        previous_path = ROOT / relative
        if previous_path.exists():
            assert hashlib.sha256(previous_path.read_bytes()).hexdigest() == digest
            checked_previous_files += 1
    records, paired, accounting = summaries(rows)
    actual = pd.read_csv(folder / 'summary.csv')
    expected = pd.DataFrame(records)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    assert json.loads((folder / 'paired.json').read_text()) == paired
    assert json.loads((folder / 'accounting.json').read_text()) == accounting
    report = {'status': 'passed', 'outer_reps': len(rows),
              'inner_attempted': accounting['inner_attempted'],
              'checks': ['source and prospective timing/protocol hashes', 'fixed scope and deterministic unique seeds',
                         'no overlap with packaged previous-study and timing-pilot stream identities',
                         'every saved bootstrap interval and rejection recomputed',
                         'six all-attempt and conditional rate rows recomputed',
                         'paired discordance and Monte Carlo intervals recomputed',
                         'failure, improper and nonconverged accounting recomputed'],
              'prior_stream_reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              'previous_checkpoint_files_available_and_verified': checked_previous_files,
              'checkpoint_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'summary_sha256': hashlib.sha256((folder / 'summary.csv').read_bytes()).hexdigest()}
    original = json.loads((folder / 'verification.json').read_text())
    assert original['status'] == 'passed' and original['no_overlap_with_previous_studies_or_pilot']
    assert original['previous_stream_ids_checked'] == len(reference['stream_ids'])
    assert original['new_stream_ids'] == len(stream_ids)
    assert original['checkpoint_sha256'] == report['checkpoint_sha256']
    assert original['summary_sha256'] == report['summary_sha256']
    (folder / 'recomputed_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(make_seed_reference() if '--prepare-seeds' in sys.argv else check(), indent=2))
