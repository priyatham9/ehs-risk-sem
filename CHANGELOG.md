# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and this project uses semantic versioning.

## [0.1.0] - 2026-09-09

### Added
- Structural equation modelling implementation in numpy for occupational safety risk with limited-information two-step estimation (measurement model first, then structural)
- Measurement module with principal-axis factoring, congeneric blocks, composite reliability, omega, alpha, AVE, Fornell-Larcker, HTMT, and factor-score determinacy
- Structural module computing path coefficients, VIF, direct/indirect/total effects from disattenuated latent correlation matrix
- Four comprehensive simulation studies showing where SEM breaks: sample-size requirements for path discrimination, equivalent-model problem, reverse-causation fitting, omitted confounders invisible in fit indices, and formative-vs-reflective misspecification
- Rare-events analysis (base-rate arithmetic, PPV, precision-recall trade-offs) at 2024 US private-industry total recordable case rate of 2.3 per 100 FTE
- Calibration, GLM (logistic regression with King-Zeng rare-event correction), and power analysis modules for coefficient precision and sample-size planning
