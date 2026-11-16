import unittest

from simulations.study_09_bootstrap_weak_measurement import choose_target, rate_record, summaries


class TestWeakMeasurementStress(unittest.TestCase):
    def test_sample_size_rule_uses_runtime_only(self):
        self.assertEqual(choose_target(100)[0], 500)
        self.assertEqual(choose_target(200)[0], 250)
        self.assertIsNone(choose_target(300)[0])
        self.assertEqual(choose_target(100)[1]['500'], 3125)

    def test_empty_conditional_rate_is_missing_not_zero(self):
        result = rate_record('percentile', 'computable_only', [])
        self.assertEqual(result['rejection_rate_denominator'], 0)
        self.assertIsNone(result['rejection_rate'])
        self.assertIsNone(result['exact_cp_low'])

    def test_failed_intervals_remain_in_all_attempt_denominators(self):
        def row(outer_valid, interval_valid, naive, percentile, failed, converged):
            return dict(observed_reject=naive, observed_valid=outer_valid,
                        observed_completed=outer_valid, observed_proper=outer_valid,
                        observed_converged=converged, bootstrap_valid=interval_valid,
                        percentile_reject=percentile, normal_reject=False,
                        bootstrap_attempted=399 if outer_valid else 0,
                        bootstrap_draws=[0.] * (399-failed) if outer_valid else [],
                        bootstrap_failed=failed, bootstrap_improper=0,
                        bootstrap_nonconverged=1 if outer_valid else 0)
        rows = [row(True, True, True, True, 0, True),
                row(True, False, True, False, 30, False),
                row(False, False, False, False, 0, False)]
        records, paired, accounting = summaries(rows)
        lookup = {(r['method'], r['conditioning']): r for r in records}
        self.assertEqual(lookup['percentile', 'all_attempts']['rejection_rate'], 1/3)
        self.assertEqual(lookup['percentile', 'computable_only']['rejection_rate'], 1)
        self.assertEqual(lookup['naive', 'computable_only']['rejection_rate_denominator'], 2)
        self.assertEqual(paired['percentile']['naive_only_reject'], 1)
        self.assertEqual(paired['percentile']['bootstrap_minus_naive'], -1/3)
        self.assertEqual(accounting['inner_planned'], 1197)
        self.assertEqual(accounting['inner_attempted'], 798)
        self.assertEqual(accounting['invalid_intervals'], 2)
        self.assertEqual(accounting['bootstrap_nonconverged'], 2)
        self.assertEqual(accounting['outer_with_bootstrap_nonconverged']['rate_denominator'], 3)


if __name__ == '__main__':
    unittest.main()
