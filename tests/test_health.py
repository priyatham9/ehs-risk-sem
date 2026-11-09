"""Data expectations and lineage: the checks must hold, seeded breakages must be caught, outputs must not drift."""

from __future__ import annotations

import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import health  # noqa: E402


class TestHealth(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = health.load()
        cls.checks = health.run_checks(cls.data)
        cls.spec = health.load_spec()

    def _failing(self, data):
        return {c["rule"] for c in health.run_checks(data) if c["status"] == "fail"}

    def _fresh(self):
        return copy.deepcopy(self.data)

    def test_no_error_severity_failures(self) -> None:
        self.assertEqual([c for c in self.checks if c["status"] == "fail"], [])

    def test_headline_counts(self) -> None:
        self.assertEqual(len(self.checks), 101)
        self.assertEqual(len(health.SCHEMA), 23)
        warns = [c["id"] for c in self.checks if c["status"] == "warn"]
        self.assertEqual(warns, ["R25", "R26", "R80"])
        rec = self.data["tables"]["study01_recovery.csv"]
        self.assertAlmostEqual(min(float(r["coverage_95_analytic"]) for r in rec), 0.7567, places=3)

    def test_check_ids_unique_and_sequential(self) -> None:
        ids = [c["id"] for c in self.checks]
        self.assertEqual(ids, ["R%02d" % (i + 1) for i in range(len(ids))])

    def test_every_results_csv_is_in_the_schema(self) -> None:
        self.assertEqual(self.data["extra_results"], [])
        for name in health.SCHEMA:
            self.assertTrue(os.path.exists(os.path.join(ROOT, "results", name)), name)

    def test_seeded_breakages_are_caught(self) -> None:
        d = self._fresh()
        T = d["tables"]
        T["study01_recovery.csv"][0]["bias"] = "0.5"
        T["study02_alert_burden.csv"][0]["ppv"] = "0.5"
        T["study02_sizing.csv"][0]["units_required"] = "1"
        T["study03_equivalent.csv"][1]["cfi"] = "0.5"
        T["fit_index_cutoffs_n200.csv"][0]["p50"] = "1e9"
        T["planner_loading075.csv"][0]["completed_fits"] = "1"
        T["study04_invariance.csv"][2]["value"] = "0.5"
        failing = self._failing(d)
        for rule in ("bias equals mean estimate minus true beta",
                     "PPV and alerts per true event recompute from Bayes' rule",
                     "units required equals events required over base rate",
                     "the two equivalent models report identical beta and fit",
                     "n200: p05 <= p50 <= p95 for every index",
                     "loading 0.75: rates are probabilities and every replication completed",
                     "composite reliabilities recompute from the listed loadings (within 0.002)"):
            self.assertIn(rule, failing)

    def test_schema_and_parse_breakages_are_caught(self) -> None:
        d = self._fresh()
        d["headers"]["study03_mediator.csv"] = ["quantity", "estimate"]
        d["tables"]["study04_roundness.csv"][0]["median"] = "0.5"
        d["tables"]["study01_bootstrap.csv"][0]["coverage_analytic"] = "nan"
        failing = self._failing(d)
        self.assertIn("study03: columns equal the declared schema (5 files)", failing)
        self.assertIn("study01: numeric columns parse as finite numbers", failing)
        self.assertIn("rounding distance is at most 0.025 and p05 <= median <= p95", failing)

    def test_synthetic_and_note_breakages_are_caught(self) -> None:
        d = self._fresh()
        com, hdr, n, rows = d["synthetic"]["rare_event.csv"]
        d["synthetic"]["rare_event.csv"] = (com, hdr, n - 1, rows[:-1])
        d["note"] = "no mention"
        failing = self._failing(d)
        self.assertTrue(any(r.startswith("rare_event.csv: labelled synthetic") for r in failing))
        self.assertIn("methods note states that it uses simulated data", failing)

    def test_extra_results_file_is_caught(self) -> None:
        d = self._fresh()
        d["extra_results"] = ["study09_new.csv"]
        self.assertIn("no results CSV outside the declared schema", self._failing(d))

    def test_declared_lineage_is_valid(self) -> None:
        self.assertEqual(health.validate_lineage(self.spec), [])
        self.assertEqual(len(self.spec["stages"]), 10)

    def test_lineage_breakages_are_caught(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["stages"][1]["function"] = "no_such_function"
        spec["stages"][2]["outputs"].append("study01")
        spec["stages"].append({"id": "loop_a", "label": "a", "script": "tools/health.py", "function": "main",
                               "command": "x", "inputs": ["lb"], "outputs": ["la"]})
        spec["stages"].append({"id": "loop_b", "label": "b", "script": "tools/health.py", "function": "main",
                               "command": "x", "inputs": ["la"], "outputs": ["lb"]})
        spec["artifacts"]["la"] = {"label": "la", "kind": "file", "paths": ["tools/health.py"]}
        spec["artifacts"]["lb"] = {"label": "lb", "kind": "file", "paths": ["tools/health.py"]}
        text = " | ".join(health.validate_lineage(spec))
        self.assertIn("no_such_function", text)
        self.assertIn("produced by both", text)
        self.assertIn("cycle among stages", text)

    def test_undeclared_committed_file_is_caught(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["artifacts"]["study01"]["paths"] = ["results/study01_recovery.csv"]
        text = " | ".join(health.validate_lineage(spec))
        self.assertIn("committed file not declared in lineage: results/study01_bootstrap.csv", text)

    def test_committed_outputs_have_not_drifted(self) -> None:
        outs, _, problems = health.outputs()
        self.assertEqual(problems, [])
        for path, text in outs.items():
            self.assertTrue(os.path.exists(path), path)
            with open(path, encoding="utf-8", newline="") as fh:
                self.assertEqual(fh.read(), text, "%s is stale; run python3 tools/health.py --write" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    unittest.main()
