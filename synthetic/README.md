# synthetic/

Everything in this directory is generated. None of it is a measurement.

Each CSV begins with `#`-prefixed header lines stating that the file is
synthetic, naming the generating model and its parameters, giving the seed, and
recording the date it was produced. Read them back with:

```python
pandas.read_csv("synthetic/survey_style.csv", comment="#")
```

Regenerate all three files:

```
python3 synthetic/generate_synthetic_datasets.py
```

The seeds are fixed, so the data rows come back bit-for-bit identical. One line
does change: the `# generated:` stamp in the header records the date the
generator was run. Verify a regeneration with

```
diff <(grep -v '^#' old.csv) <(grep -v '^#' new.csv)
```

which compares the data and ignores the header.

## Why these files exist

Two reasons, both narrow.

1. The worked example in the top-level `README.md` and the integration tests
   need a dataset that does not change between runs.
2. Some of the arguments this repository makes are about what an estimator does
   when the truth is known. That comparison is only possible on generated data,
   which is why the survey and rare-event files carry the generating latent
   scores and the generating probabilities alongside the observed columns.
   Those columns do not exist in real data and are labelled as such.

## What they are not

They are not a substitute for data, they carry no information about workplace
injuries, and no number computed from them appears as a result anywhere in this
repository except as a demonstration explicitly labelled as simulation.

Real empirical work on injury data belongs to real public sources: OSHA
Injury Tracking Application filings, BLS Survey of Occupational Injuries and
Illnesses estimates, MSHA accident records. This repository does not analyse
any of them. The one incidence rate it uses as an input constant is cited and
verified against the primary BLS source at its definition in
`ehs_risk_sem/rare_events.py`.

## Files

| File | Rows | What it is |
| --- | --- | --- |
| `survey_style.csv` | 1,200 | Five latents, three congeneric indicators each, loadings 0.75, continuous outcome. The generating structural paths are 0.45, 0.30, -0.25, -0.20. Includes `latent_*` columns holding the generating latent scores. |
| `rare_event.csv` | 25,000 | Twelve indicators of four latent predictors plus a binary outcome at a marginal rate of 0.005. Includes `true_probability`, the generating probability per row. |
| `zero_inflated_counts.csv` | 8,000 | The same predictors with a zero-inflated Poisson count outcome: a structural-zero process at rate 0.38 mixed with Poisson counts of mean 1.5. Includes the generating Poisson mean and structural-zero probability. |
