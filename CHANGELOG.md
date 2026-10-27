# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [Unreleased]

### Added
- `.zenodo.json` and `codemeta.json` (title, description, creators from `CITATION.cff`, MIT licence, keywords, related identifiers for the sibling repositories and the programme site; no DOI or ORCID) so a release is archived with correct metadata once Zenodo is linked; `CITATION.cff` gains `keywords`. `tools/check_citation.py` (stdlib, run by `.github/workflows/citation.yml`) checks the required CFF 1.2.0 fields and that `CITATION.cff`, `.zenodo.json`, `codemeta.json` and `LICENSE` agree on title, authors, keywords and licence. No release or tag created.
- `tools/sample_size_planner.py`: Monte Carlo sample-size planner for SEM safety-climate surveys (proper-solution rate, loading and path stability by N, recommended N), with committed tables `results/planner_loading075.csv` and `results/planner_loading050.csv` and `tests/test_sample_size_planner.py`
- `docs/methods-note.md`: methods note for safety practitioners on when SEM fit indices pass while path estimates are too imprecise to rank or use as weights, with sample-size guidance and verified references
- `tools/methods_note_tables.py`: builds the note's tables from the committed `results/` CSV files
- `tests/test_methods_note.py`: fails if the note's tables or quoted numbers drift from `results/`

### Changed
- README test count updated from 114 to 119 to match the test suite

## [0.1.0] - 2026-09-09

### Added
- Structural equation modelling implementation in numpy for occupational safety risk with limited-information two-step estimation (measurement model first, then structural)
- Measurement module with principal-axis factoring, congeneric blocks, composite reliability, omega, alpha, AVE, Fornell-Larcker, HTMT, and factor-score determinacy
- Structural module computing path coefficients, VIF, direct/indirect/total effects from disattenuated latent correlation matrix
- Four comprehensive simulation studies showing where SEM breaks: sample-size requirements for path discrimination, equivalent-model problem, reverse-causation fitting, omitted confounders invisible in fit indices, and formative-vs-reflective misspecification
- Rare-events analysis (base-rate arithmetic, PPV, precision-recall trade-offs) at 2024 US private-industry total recordable case rate of 2.3 per 100 FTE
- Calibration, GLM (logistic regression with King-Zeng rare-event correction), and power analysis modules for coefficient precision and sample-size planning
