"""The methods note quotes only numbers that are in the committed results files.

``docs/methods-note.md`` holds generated tables between marker comments, and
quotes further numbers in its prose. These tests rebuild the tables from
``results/`` and check each quoted number, in the phrase where it appears,
against the value computed from ``results/``.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import methods_note_tables as mnt  # noqa: E402

# Phrase in the note -> the key_facts() entries it contains.
PHRASES = [
    "reaches 80% power\nat N = {n_close_fit}",
    "standard error of 0.05 needs\nN = {n_se_05}",
    "one of 0.30 needs N = {n_045_vs_030}",
    "-0.25 from -0.20 needs N = {n_025_vs_020}",
    "between {cov_analytic_min} and\n{cov_analytic_max}",
    "restored coverage to {cov_boot_min} to {cov_boot_max} at N = 500",
    "within about {half_width_n100} of the truth",
    "the half-width is about\n{half_width_n200} and at N = 500 about {half_width_n500}",
    "between {se_ratio_min_pct}% and {se_ratio_max_pct}% of the empirical variability",
    "cover the true value only {cov_analytic_min} to {cov_analytic_max} of the time",
    "no more than {max_n_large_gaps}\nobservations",
    "standard error of any difference is {se_diff_n250}",
    "N = 500, {boot_reps} replications.",
    "ranged from {cov_boot_min} to {cov_boot_max}, against {boot_analytic_min} to "
    "{boot_analytic_max}",
    "about {boot_mc_se_pts} percentage points",
    "{boot_wider_min_pct}% to {boot_wider_max_pct}% wider",
    "median of {sum_abs_median} and a 5th to 95th\npercentile range of {sum_abs_p05} to "
    "{sum_abs_p95}",
    "A difference of 0.05 needs about {n_025_vs_020}",
]


def _norm(text: str) -> str:
    return " ".join(text.split())


class TestMethodsNote(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(mnt.NOTE, encoding="utf-8") as fh:
            cls.text = fh.read()
        cls.facts = mnt.key_facts()

    def test_all_tables_present(self) -> None:
        self.assertEqual(sorted(mnt.blocks_in(self.text)), sorted(mnt.TABLES))

    def test_tables_match_results(self) -> None:
        self.assertEqual(
            mnt.render(self.text),
            self.text,
            "docs/methods-note.md is stale: run python3 tools/methods_note_tables.py",
        )

    def test_quoted_numbers_match_results(self) -> None:
        body = _norm(self.text)
        for phrase in PHRASES:
            expected = _norm(phrase.format(**self.facts))
            with self.subTest(phrase=phrase):
                self.assertIn(expected, body)

    def test_no_em_or_en_dashes(self) -> None:
        self.assertNotIn("—", self.text)
        self.assertNotIn("–", self.text)

    def test_abstract_requirements_agree_with_table(self) -> None:
        abstract = self.text.split("## 1. Problem")[0]
        for key in ("n_close_fit", "n_se_05", "n_045_vs_030", "n_025_vs_020"):
            with self.subTest(key=key):
                self.assertIn(f"N = {self.facts[key]}", _norm(abstract))


if __name__ == "__main__":
    unittest.main()
