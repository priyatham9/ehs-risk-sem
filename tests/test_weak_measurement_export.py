import json
from pathlib import Path
import tempfile
import unittest

from tools.export_weak_measurement_audit import read_summary


class TestWeakMeasurementExport(unittest.TestCase):
    def test_csv_precision_and_missing_conditional_values_are_preserved(self):
        content = (
            'method,conditioning,rejection_rate,rejection_rate_count,rejection_rate_denominator,'
            'rejection_rate_mcse,rejection_rate_low,rejection_rate_high,exact_cp_low,exact_cp_high\n'
            'percentile,all_attempts,0.052,26,500,0.00992935043192655,'
            '0.035730265115705276,0.07510114393784553,0.03424569808276731,0.07526637103062303\n'
            'percentile,computable_only,,0,0,,,,,\n'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'summary.csv'
            path.write_text(content)
            rows = read_summary(path)
        self.assertEqual(str(rows[0]['rejection_rate_mcse']), '0.00992935043192655')
        self.assertIsInstance(rows[0]['rejection_rate_count'], int)
        self.assertIsNone(rows[1]['rejection_rate'])
        self.assertIsNone(rows[1]['exact_cp_low'])
        self.assertEqual(json.loads(json.dumps(rows)), rows)


if __name__ == '__main__':
    unittest.main()
