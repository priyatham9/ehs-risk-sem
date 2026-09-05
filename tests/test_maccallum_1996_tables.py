"""External validation of :mod:`ehs_risk_sem.power` against MacCallum, Browne &
Sugawara (1996), *Psychological Methods*, 1(2), 130-149.

This is the one test in the suite that checks the package against numbers
printed in a peer-reviewed paper rather than against a closed form, a Monte
Carlo draw, or itself. The values below were transcribed from the text of the
published article (Tables 2, 4 and 5), not from a secondary reproduction, and
were re-checked cell by cell against the article PDF during an independent
verification pass (every row of all three tables, including the rows that were
originally omitted).

Anchors quoted verbatim in the article's prose, which serve as a check that the
transcription itself is right:

* "with d = 40 and N = 200, the probability of rejecting H0: e <= 0.05 is
  approximately .69 if ea = 0.08"  (Table 2)
* "for d = 40, Nmin = 252 to assure power of at least 0.80"  (Table 4)
* "with d = 100, a power of 0.80 for the test of close fit ... is achieved
  with N = 132"  (Table 4)
* "with d = 50 and N >= 243, the likelihood of rejecting the hypothesis of
  exact fit would be at least .80 ... power would be greater than 0.50 with
  N >= 148"  (Table 5)

Tolerances. Power values are compared at 0.001, the precision at which the
paper prints them. Sample sizes are compared at the looser of one observation
or 0.25% of the published value: MacCallum et al. locate Nmin by
interval-halving to a stated approximation rather than exactly, so the last
unit of N is a property of their search convention, not of the procedure. The
only cells needing more than one unit of slack are d = 2 and d = 4, where Nmin
is in the thousands.
"""

from __future__ import annotations

import unittest

from ehs_risk_sem.power import min_n_for_rmsea_power, rmsea_power

# Table 4: minimum N for power 0.80. df -> (test of close fit, test of
# not-close fit). Close fit is e0 = 0.05 against ea = 0.08; not-close fit is
# e0 = 0.05 against ea = 0.01. alpha = .05 throughout.
TABLE_4 = {
    2: (3488, 2382), 4: (1807, 1426), 6: (1238, 1069), 8: (954, 875),
    10: (782, 750), 12: (666, 663), 14: (585, 598), 16: (522, 547),
    18: (472, 508), 20: (435, 474), 25: (363, 411), 30: (314, 366),
    35: (279, 333), 40: (252, 307), 45: (231, 286), 50: (214, 268),
    55: (200, 253), 60: (187, 240), 65: (177, 229), 70: (168, 219),
    75: (161, 210), 80: (154, 202), 85: (147, 195), 90: (142, 189),
    95: (136, 183), 100: (132, 178),
}

# Table 5: minimum N for the test of exact fit (e0 = 0.00, ea = 0.05).
# df -> (N for power 0.80, N for power 0.50).
TABLE_5 = {
    2: (1926, 994), 4: (1194, 644), 6: (910, 502), 8: (754, 422),
    10: (651, 369), 12: (579, 332), 14: (525, 304), 16: (483, 280),
    18: (449, 262), 20: (421, 247), 25: (368, 218), 30: (329, 196),
    35: (300, 180), 40: (277, 167), 45: (258, 157), 50: (243, 148), 55: (230, 140),
    60: (218, 134), 65: (209, 128), 70: (200, 123), 75: (193, 119),
    80: (186, 115), 85: (179, 111), 90: (174, 108), 95: (168, 105),
    100: (164, 102),
}

# Table 2: power at selected df and N, in the order (close, not close, exact).
# Columns are N = 100, 200, 300, 400, 500.
TABLE_2 = {
    5: ((0.127, 0.199, 0.269, 0.335, 0.397),
        (0.081, 0.124, 0.181, 0.248, 0.324),
        (0.112, 0.188, 0.273, 0.362, 0.449)),
    10: ((0.169, 0.294, 0.413, 0.520, 0.612),
         (0.105, 0.191, 0.304, 0.429, 0.555),
         (0.141, 0.266, 0.406, 0.541, 0.661)),
    15: ((0.206, 0.378, 0.533, 0.661, 0.760),
         (0.127, 0.254, 0.414, 0.578, 0.720),
         (0.167, 0.336, 0.516, 0.675, 0.797)),
    20: ((0.241, 0.454, 0.633, 0.766, 0.855),
         (0.148, 0.314, 0.513, 0.695, 0.830),
         (0.192, 0.400, 0.609, 0.773, 0.882)),
    30: ((0.307, 0.585, 0.780, 0.893, 0.951),
         (0.187, 0.424, 0.673, 0.850, 0.943),
         (0.237, 0.512, 0.750, 0.894, 0.962)),
    40: ((0.368, 0.688, 0.872, 0.954, 0.985),
         (0.224, 0.523, 0.788, 0.930, 0.982),
         (0.279, 0.606, 0.843, 0.952, 0.988)),
    50: ((0.424, 0.769, 0.928, 0.981, 0.995),
         (0.261, 0.608, 0.866, 0.969, 0.995),
         (0.319, 0.684, 0.903, 0.979, 0.997)),
    60: ((0.477, 0.831, 0.960, 0.992, 0.999),
         (0.296, 0.681, 0.917, 0.987, 0.999),
         (0.356, 0.748, 0.941, 0.991, 0.999)),
    70: ((0.525, 0.877, 0.978, 0.997, 1.000),
         (0.330, 0.743, 0.949, 0.994, 1.000),
         (0.393, 0.801, 0.965, 0.996, 1.000)),
    80: ((0.570, 0.911, 0.988, 0.999, 1.000),
         (0.363, 0.794, 0.970, 0.998, 1.000),
         (0.427, 0.843, 0.979, 0.998, 1.000)),
    90: ((0.612, 0.937, 0.994, 1.000, 1.000),
         (0.395, 0.836, 0.982, 0.999, 1.000),
         (0.460, 0.877, 0.988, 0.999, 1.000)),
    100: ((0.650, 0.955, 0.997, 1.000, 1.000),
          (0.426, 0.870, 0.990, 1.000, 1.000),
          (0.491, 0.904, 0.993, 1.000, 1.000)),
}

SAMPLE_SIZES = (100, 200, 300, 400, 500)


def _n_tolerance(published: int) -> float:
    """One observation, or 0.25% of N, whichever is looser."""
    return max(1.0, 0.0025 * published)


class TestPublishedCellCount(unittest.TestCase):
    """The README quotes how many published cells this file pins.

    Asserted here so the two cannot drift apart.
    """

    def test_cell_count(self) -> None:
        n_table_2 = sum(len(col) for rows in TABLE_2.values() for col in rows)
        n_table_4 = 2 * len(TABLE_4)
        n_table_5 = 2 * len(TABLE_5)
        self.assertEqual(n_table_2, 180)
        self.assertEqual(n_table_4, 52)
        self.assertEqual(n_table_5, 52)
        self.assertEqual(n_table_2 + n_table_4 + n_table_5, 284)


class TestTable2Power(unittest.TestCase):
    """Power values, printed to three decimals in the paper."""

    def _check(self, rmsea_null: float, rmsea_alt: float, row: int) -> None:
        for df, rows in TABLE_2.items():
            for n, published in zip(SAMPLE_SIZES, rows[row]):
                got = rmsea_power(df, n, rmsea_null, rmsea_alt)
                self.assertAlmostEqual(
                    got, published, delta=0.001,
                    msg=f"df={df}, N={n}: published {published}, computed {got:.4f}",
                )

    def test_close_fit_column(self) -> None:
        self._check(0.05, 0.08, 0)

    def test_not_close_fit_column(self) -> None:
        self._check(0.05, 0.01, 1)

    def test_exact_fit_column(self) -> None:
        self._check(0.0, 0.05, 2)

    def test_prose_anchor_df40_n200(self) -> None:
        """"with d = 40 and N = 200 ... approximately .69"."""
        self.assertAlmostEqual(rmsea_power(40, 200, 0.05, 0.08), 0.69, delta=0.005)


class TestTable4MinimumSampleSize(unittest.TestCase):
    def test_close_fit_column(self) -> None:
        for df, (published, _) in TABLE_4.items():
            got = min_n_for_rmsea_power(df, 0.80, 0.05, 0.08)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertLessEqual(
                abs(got - published), _n_tolerance(published),
                msg=f"df={df}: published {published}, computed {got}",
            )

    def test_not_close_fit_column(self) -> None:
        for df, (_, published) in TABLE_4.items():
            got = min_n_for_rmsea_power(df, 0.80, 0.05, 0.01)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertLessEqual(
                abs(got - published), _n_tolerance(published),
                msg=f"df={df}: published {published}, computed {got}",
            )

    def test_prose_anchors(self) -> None:
        """"for d = 40, Nmin = 252" and "with d = 100 ... N = 132"."""
        self.assertEqual(min_n_for_rmsea_power(40, 0.80, 0.05, 0.08), 252)
        self.assertEqual(min_n_for_rmsea_power(100, 0.80, 0.05, 0.08), 132)

    def test_crossover_at_df_14(self) -> None:
        """"At low values of d, Nmin for the test of close fit is larger than
        Nmin for the test of not-close fit. For d > 14, the relationship is
        reversed." Checked as a property of the computed values, not the table.
        """
        for df in (2, 4, 6, 8, 10, 12):
            close = min_n_for_rmsea_power(df, 0.80, 0.05, 0.08)
            not_close = min_n_for_rmsea_power(df, 0.80, 0.05, 0.01)
            assert close is not None and not_close is not None
            self.assertGreater(close, not_close, msg=f"df={df}")
        for df in (16, 20, 40, 100):
            close = min_n_for_rmsea_power(df, 0.80, 0.05, 0.08)
            not_close = min_n_for_rmsea_power(df, 0.80, 0.05, 0.01)
            assert close is not None and not_close is not None
            self.assertLess(close, not_close, msg=f"df={df}")


class TestTable5ExactFit(unittest.TestCase):
    def test_power_080_column(self) -> None:
        for df, (published, _) in TABLE_5.items():
            got = min_n_for_rmsea_power(df, 0.80, 0.0, 0.05)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertLessEqual(
                abs(got - published), _n_tolerance(published),
                msg=f"df={df}: published {published}, computed {got}",
            )

    def test_power_050_column(self) -> None:
        for df, (_, published) in TABLE_5.items():
            got = min_n_for_rmsea_power(df, 0.50, 0.0, 0.05)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertLessEqual(
                abs(got - published), _n_tolerance(published),
                msg=f"df={df}: published {published}, computed {got}",
            )

    def test_prose_anchor_df50(self) -> None:
        """"with d = 50 and N >= 243 ... at least .80 ... greater than 0.50
        with N >= 148"."""
        self.assertEqual(min_n_for_rmsea_power(50, 0.80, 0.0, 0.05), 243)
        self.assertEqual(min_n_for_rmsea_power(50, 0.50, 0.0, 0.05), 148)


class TestReadmeSampleSizeCell(unittest.TestCase):
    """The README quotes N = 153 for df = 80. Table 4 prints 154.

    Both are right under their own convention: power at N = 153 is 0.8002,
    so 153 is the smallest integer attaining 0.80, and MacCallum et al. locate
    Nmin by interval-halving to an approximation. Pinned here so the one-unit
    gap stays documented rather than becoming a silent discrepancy.
    """

    def test_df80_boundary(self) -> None:
        self.assertEqual(min_n_for_rmsea_power(80, 0.80, 0.05, 0.08), 153)
        self.assertGreaterEqual(rmsea_power(80, 153, 0.05, 0.08), 0.80)
        self.assertLess(rmsea_power(80, 152, 0.05, 0.08), 0.80)
        self.assertEqual(TABLE_4[80][0], 154)


if __name__ == "__main__":
    unittest.main()
