/* Completed simulation summaries are the only source of plotted results.
   No fitting, resampling, power interpolation, or inference is done in-browser. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const percent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
  const number = (value) => Number(value).toLocaleString("en-US");
  const decimal = (value, digits = 3) => Number(value).toFixed(digits);
  const signed = (value) =>
    `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(2)}`;
  const state = {
    pair: "positive",
    loading: 0.75,
    multiplier: 1,
    metric: "calibrated",
    bootstrapMethod: "percentile",
    uncertainty: true,
  };
  const methods = [
    {
      id: "oracle",
      name: "Oracle OLS",
      subtitle: "Knows the latent scores and population units",
      key: "oracle_rejection_rate",
    },
    {
      id: "naive",
      name: "Two-step, naive uncertainty",
      subtitle: "Plug-in SE omits measurement estimation",
      key: "observed_naive_rejection_rate",
    },
    {
      id: "calibrated",
      name: "Null-calibrated diagnostic",
      subtitle: "Critical value learned from an independent simulated null",
      key: "observed_calibrated_rejection_rate",
    },
  ];
  const metricInfo = {
    calibrated: {
      label: "Calibrated rejection under the alternative",
      description:
        "How often the simulation-calibrated procedure rejects zero when the specified difference exists. This diagnostic uses the known generating model. It is not an implementable real-data correction.",
      stage: "alternative",
      key: "observed_calibrated_rejection_rate",
      calibrated: true,
    },
    "false-positive": {
      label: "Naive false-positive rate",
      description:
        "How often the naive two-step test rejects zero when the compared coefficients are actually equal. The nominal reference is 5%. A larger value is false-positive inflation, not added power.",
      stage: "null_validation",
      key: "observed_naive_rejection_rate",
    },
    reversal: {
      label: "Wrong ordering among good fits",
      description:
        "The fraction of valid, conventionally well-fitting alternative simulations whose estimated coefficient contrast has the opposite sign to the known truth. This denominator is conditional on good fit.",
      stage: "alternative",
      key: "wrong_ranking_given_good_fit",
    },
  };
  let audit, rows, toastTimer;
  const pairLabel = () =>
    state.pair === "positive" ? "Positive pair" : "Negative pair";
  function announce(message) {
    $("live-status").textContent = message;
  }
  function toast(message) {
    $("toast").textContent = message;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      $("toast").hidden = true;
    }, 3200);
  }
  function row(stage, loading = state.loading, multiplier = state.multiplier) {
    return rows.find(
      (r) =>
        r.pair === state.pair &&
        r.loading === loading &&
        r.multiplier === multiplier &&
        r.stage === stage,
    );
  }
  function fisherRow() {
    return audit.fisher.rows.find(
      (r) =>
        r.pair === state.pair &&
        r.loading === state.loading &&
        r.stage === "alternative",
    );
  }
  function bounds(r, key, calibrated = false) {
    return calibrated
      ? [
          r.calibrated_two_sample_bootstrap_low,
          r.calibrated_two_sample_bootstrap_high,
        ]
      : [r[`${key}_low`], r[`${key}_high`]];
  }
  function intervalText(r, key, calibrated = false) {
    const [low, high] = bounds(r, key, calibrated);
    return `${percent(low)}–${percent(high)}`;
  }
  function counts(r, key) {
    return `${number(r[`${key}_count`])} / ${number(r[`${key}_denominator`])}`;
  }
  function figure(r, key, method, target, calibrated = false) {
    const [low, high] = bounds(r, key, calibrated),
      value = r[key];
    return `<div class="rate-figure ${method}"><div class="rate-numbers"><strong class="rate-value">${percent(value)}</strong><span class="rate-count">${counts(r, key)}</span></div><div class="rate-track" aria-hidden="true" style="--point:${value * 100}%;--low:${low * 100}%;--high:${high * 100}%;--target:${target * 100}%"><i class="reference-line"></i><i class="interval-bar"></i><i class="rate-dot"></i></div><div class="interval-label">95% MC ${percent(low)}–${percent(high)}</div></div>`;
  }
  function readUrl() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (["positive", "negative"].includes(p.get("pair")))
      state.pair = p.get("pair");
    if ([0.5, 0.75, 0.9].includes(Number(p.get("loading"))))
      state.loading = Number(p.get("loading"));
    if ([0.5, 1, 2].includes(Number(p.get("size"))))
      state.multiplier = Number(p.get("size"));
    if (Object.hasOwn(metricInfo, p.get("metric")))
      state.metric = p.get("metric");
    if (["percentile", "normal"].includes(p.get("bootstrap")))
      state.bootstrapMethod = p.get("bootstrap");
  }
  function writeUrl() {
    const p = new URLSearchParams({
      pair: state.pair,
      loading: state.loading,
      size: state.multiplier,
      metric: state.metric,
      bootstrap: state.bootstrapMethod,
    });
    history.replaceState(
      null,
      "",
      `${location.pathname}${location.search}#${p}`,
    );
  }
  function renderControls() {
    for (const [attr, key] of [
      ["pair", "pair"],
      ["loading", "loading"],
      ["multiplier", "multiplier"],
    ])
      document
        .querySelectorAll(`[data-${attr}]`)
        .forEach((button) =>
          button.setAttribute(
            "aria-pressed",
            String(String(state[key]) === button.dataset[attr]),
          ),
        );
    document.querySelectorAll("[data-n-label]").forEach((el) => {
      el.textContent = `N ${number(row("alternative", state.loading, Number(el.dataset.nLabel)).n)}`;
    });
    const alt = row("alternative");
    $("case-label").textContent =
      `${pairLabel()} · Δ ${signed(alt.true_difference)} · λ ${state.loading.toFixed(2)} · N ${number(alt.n)}`;
    $("case-detail").textContent =
      `${number(alt.attempts)} independent surveys per evaluation stream · two-sided tests`;
    $("comparison-description").textContent =
      `Selected condition: ${alt.cell}. All attempted surveys remain in each rejection-rate denominator.`;
  }
  function renderComparison() {
    const alt = row("alternative"),
      nul = row("null_validation");
    $("comparison-rows").innerHTML = methods
      .map(
        (m) =>
          `<div class="comparison-row ${m.id}" role="row"><div class="method-label" role="rowheader"><div class="method-name"><i class="method-dot" aria-hidden="true"></i>${m.name}</div><p class="method-subtitle">${m.subtitle}</p></div><div role="cell" aria-label="${esc(m.name)}: alternative rejection ${percent(alt[m.key])}; 95% Monte Carlo interval ${intervalText(alt, m.key, m.id === "calibrated")}; ${counts(alt, m.key)}">${figure(alt, m.key, m.id, 0.8, m.id === "calibrated")}</div><div role="cell" aria-label="${esc(m.name)}: null false positives ${percent(nul[m.key])}; 95% Monte Carlo interval ${intervalText(nul, m.key, m.id === "calibrated")}; ${counts(nul, m.key)}">${figure(nul, m.key, m.id, 0.05, m.id === "calibrated")}</div></div>`,
      )
      .join("");
    const ratio = alt.observed_empirical_sd / alt.observed_mean_naive_se;
    $("case-reading").innerHTML =
      `<span class="reading-label">READING THIS CONDITION</span>The naive two-step test rejects <strong>${percent(nul.observed_naive_rejection_rate)}</strong> of held-out nulls against a nominal 5% reference. With a separate simulated null cutoff, its alternative rejection is <strong>${percent(alt.observed_calibrated_rejection_rate)}</strong>, compared with <strong>${percent(alt.oracle_rejection_rate)}</strong> for the latent-score oracle.<br><span class="reading-detail">Across alternative surveys, the contrast’s empirical SD is <strong>${ratio.toFixed(2)}×</strong> its mean naive SE (${decimal(alt.observed_empirical_sd)} versus ${decimal(alt.observed_mean_naive_se)}). The calibrated critical value is ${decimal(alt.null_calibrated_critical_z)}, rather than 1.960.</span>`;
    $("fit-rate").textContent = percent(alt.fit_good_rate);
    $("fit-interval").textContent =
      `${counts(alt, "fit_good_rate")} · 95% MC ${intervalText(alt, "fit_good_rate")}`;
    $("reversal-rate").textContent = percent(alt.wrong_ranking_given_good_fit);
    $("reversal-interval").textContent =
      `${counts(alt, "wrong_ranking_given_good_fit")} · 95% MC ${intervalText(alt, "wrong_ranking_given_good_fit")}`;
    const flagged = alt.attempts - alt.observed_converged_rate_count;
    $("fit-accounting").innerHTML =
      `Alternative stream: ${number(alt.observed_completed_rate_count)}/${number(alt.attempts)} fits completed; ${number(alt.observed_proper_rate_count)} proper solutions; ${number(alt.observed_converged_rate_count)} converged. ${flagged ? `<span class="flag">${number(flagged)} flagged nonconverged fits are retained in the reported rates.</span> The complete artifact includes a convergence sensitivity check.` : "Convergence status is recorded separately from fit quality and inference."}`;
  }
  function renderMap() {
    const info = metricInfo[state.metric];
    document.querySelectorAll("[data-metric]").forEach((button) => {
      const selected = button.dataset.metric === state.metric;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("map-metric-description").textContent = info.description;
    $("map-contrast").textContent =
      `${pairLabel()} · Δ ${signed(row("alternative").true_difference)}`;
    $("design-map").setAttribute(
      "aria-label",
      `${info.label}, by loading and sample size, ${pairLabel()}`,
    );
    const sizes = [0.5, 1, 2],
      loadings = [0.5, 0.75, 0.9];
    let html = '<div class="map-axis-corner">LOADING ↓<br>SAMPLE N →</div>';
    for (const size of sizes)
      html += `<div class="map-column-head">${size === 0.5 ? "½" : size}× N₀<span>N = ${number(row("alternative", state.loading, size).n)}</span></div>`;
    for (const loading of loadings) {
      html += `<div class="map-row-head">${loading.toFixed(2)}<span>${loading === 0.5 ? "Weaker" : loading === 0.75 ? "Moderate" : "Stronger"}</span></div>`;
      for (const size of sizes) {
        const r = row(info.stage, loading, size),
          value = r[info.key],
          selected = loading === state.loading && size === state.multiplier;
        const shade = Math.round(255 - value * 30),
          green = Math.round(255 - value * 18),
          blue = Math.round(255 - value * 7);
        html += `<button class="map-cell" data-map-loading="${loading}" data-map-size="${size}" aria-pressed="${selected}" aria-label="${info.label}: ${percent(value)}, 95% Monte Carlo interval ${intervalText(r, info.key, info.calibrated)}, loading ${loading}, N ${number(r.n)}. Select condition." style="--cell-bg:rgb(${shade},${green},${blue});--point:${value * 100}%">${selected ? '<span class="selected-marker">SELECTED</span>' : ""}<span class="map-value">${percent(value)}</span><span class="map-interval">${intervalText(r, info.key, info.calibrated)}</span><span class="map-count">${counts(r, info.key)}</span><div class="cell-track" aria-hidden="true"><i></i></div></button>`;
      }
    }
    $("design-map").innerHTML = html;
    $("map-caption-detail").textContent =
      `Each cell: rate, 95% Monte Carlo interval, and event count / denominator. ${state.metric === "reversal" ? "Denominator: valid, good-fit surveys." : "Denominator: all attempted evaluation surveys."}`;
  }
  function renderFisher() {
    const f = fisherRow(),
      v = audit.fisher.validation.find(
        (r) => r.pair === state.pair && r.loading === state.loading,
      );
    $("variance-inflation").textContent =
      `${f.variance_inflation_vs_standardized_latent.toFixed(2)}×`;
    $("fisher-reading").textContent =
      `At loading ${state.loading.toFixed(2)}, the full-information contrast variance is ${f.variance_inflation_vs_standardized_latent.toFixed(2)} times the standardized-latent reference. Even an asymptotically efficient estimator has to pay for imperfect measurement under these assumptions.`;
    $("fisher-condition").textContent =
      `${pairLabel()} · λ ${state.loading.toFixed(2)}`;
    $("advertised-n").textContent = number(f.advertised_n);
    $("fisher-n").textContent = number(f.asymptotic_n_for_80pct_power);
    const max = Math.max(f.advertised_n, f.asymptotic_n_for_80pct_power);
    $("advertised-bar").style.width = `${(f.advertised_n / max) * 100}%`;
    $("fisher-bar").style.width =
      `${(f.asymptotic_n_for_80pct_power / max) * 100}%`;
    $("fisher-power-base").textContent = percent(
      f.asymptotic_power_at_advertised_n,
    );
    $("fisher-power-double").textContent = percent(
      f.asymptotic_power_at_double_n,
    );
    $("fisher-validation").innerHTML =
      `<p>The expected Gaussian Fisher information is calculated for a freely estimated congeneric covariance model with ${number(audit.fisher.manifest.free_parameters)} parameters. The target is the standardized structural coefficient contrast, not a raw regression coefficient. Reported sample sizes use a two-sided asymptotic Wald calculation with 5% nominal significance and 80% power.</p><p>For this selection, <strong>N × contrast variance = ${decimal(f.full_information_variance_times_n, 6)}</strong>. The independently computed marker-parameterization result is <strong>${decimal(v.numerical_marker_parameterization, 6)}</strong>; relative difference <code>${v.relative_difference.toExponential(2)}</code>. Information rank: ${v.information_rank}. This cross-check validates the calculation, not the real-world model assumptions.</p><p>The benchmark always reports N₀ and 2N₀ for the selected contrast and loading. It is not interpolated to untested sample sizes, and it is not a finite-sample guarantee.</p>`;
  }
  function renderBootstrap() {
    const boot = audit.bootstrap;
    $("bootstrap-unavailable").hidden = boot.complete;
    $("bootstrap-results").hidden = !boot.complete;
    if (!boot.complete) {
      const m = boot.manifest;
      $("bootstrap-design").textContent = m
        ? `Planned design: ${number(m.outer_reps)} outer surveys × 4 conditions; ${number(m.inner_reps)} row-resampling refits per survey. No partial result is displayed.`
        : "No completed follow-up artifact is available.";
      return;
    }
    const alt = boot.rows.find(
      (r) => r.pair === state.pair && r.stage === "alternative",
    );
    const nul = boot.rows.find(
      (r) => r.pair === state.pair && r.stage === "null_validation",
    );
    $("bootstrap-method").value = state.bootstrapMethod;
    $("bootstrap-case").textContent =
      `${pairLabel()} · loading ${alt.loading.toFixed(2)} · N ${number(alt.n)} · ${number(alt.attempts)} outer surveys per stage`;
    const key = `${state.bootstrapMethod}_reject_rate`,
      label =
        state.bootstrapMethod === "percentile"
          ? "Percentile bootstrap"
          : "Normal + bootstrap SE";
    for (const [element, r, target] of [
      ["bootstrap-null", nul, 0.05],
      ["bootstrap-alternative", alt, 0.8],
    ]) {
      $(element).innerHTML =
        `<div class="bootstrap-result-row naive"><div class="method-name"><i class="method-dot" aria-hidden="true"></i>Naive two-step</div>${figure(r, "observed_reject_rate", "naive", target)}</div><div class="bootstrap-result-row calibrated"><div class="method-name"><i class="method-dot" aria-hidden="true"></i>${label}</div>${figure(r, key, "calibrated", target)}</div><div class="bootstrap-result-row oracle"><div class="method-name"><i class="method-dot" aria-hidden="true"></i>Oracle OLS</div>${figure(r, "oracle_reject_rate", "oracle", target)}</div>`;
    }
    const coverage = `${state.bootstrapMethod}_coverage_rate`;
    const aboveNominal = nul[`${key}_low`] > 0.05;
    $("bootstrap-reading").classList.toggle(
      "calibration-concern",
      aboveNominal,
    );
    $("bootstrap-reading").innerHTML =
      `<span class="reading-label">THE NULL CHECK STILL APPLIES</span>${label} rejects <strong>${percent(nul[key])}</strong> of null surveys against the 5% reference (95% pointwise Wilson MC interval ${intervalText(nul, key)}), compared with ${percent(nul.observed_reject_rate)} for the naive test. ${aboveNominal ? '<strong>This is a residual calibration concern.</strong> The Wilson interval narrowly excludes 5%, while the exact-interval and multiple-comparison checks in the <a href="model-audit-methods.md">methods note</a> temper the evidence for inflation. The alternative result remains a rejection rate, not demonstrated 5%-level power.' : "The interval includes the 5% reference. It does not establish exact calibration or performance outside this simulated condition."}`;
    const check = audit.bootstrap.calibration_check;
    $("bootstrap-calibration-detail").hidden =
      state.pair !== "negative" || !check;
    if (check && state.pair === "negative")
      $("bootstrap-calibration-detail-body").innerHTML =
        `<p>The original percentile procedure rejected ${number(check.successes)} of ${number(check.attempts)} negative-pair null surveys. Its 95% Wilson interval is <strong>${percent(check.wilson_low, 2)}–${percent(check.wilson_high, 2)}</strong>; the exact equal-tailed Clopper–Pearson interval is <strong>${percent(check.exact_cp_low, 2)}–${percent(check.exact_cp_high, 2)}</strong>. The exact interval includes the 5% reference.</p><p>Probability-ordering exact two-sided p = ${check.two_sided_probability_ordering_p.toFixed(4)}. Bonferroni across the two original null conditions gives p = ${check.bonferroni_two_original_null_conditions_p.toFixed(4)}. That adjustment covers these two conditions, not all analyses.</p><p>The exact interval is equal-tailed, while the two-sided p-value orders discrete outcomes by probability. Their threshold decisions need not agree. This is a calibration concern requiring independent validation, not proof of a universal defect or a certified correction.</p>`;
    $("bootstrap-coverage").textContent = percent(alt[coverage]);
    $("bootstrap-coverage-interval").textContent =
      `${counts(alt, coverage)} · 95% MC ${intervalText(alt, coverage)} · nominal 95% coverage`;
    $("bootstrap-accounting").textContent =
      `This contrast: ${number(alt.bootstrap_attempts + nul.bootstrap_attempts)} inner refits attempted; ${number(alt.bootstrap_failed + nul.bootstrap_failed)} failed; ${number(alt.bootstrap_improper + nul.bootstrap_improper)} improper; ${number(alt.bootstrap_nonconverged + nul.bootstrap_nonconverged)} nonconverged. Computable outer intervals: ${number(alt.bootstrap_valid_rate_count)}/${number(alt.attempts)} under the alternative and ${number(nul.bootstrap_valid_rate_count)}/${number(nul.attempts)} under the null. Failed or noncomputable outer intervals remain non-rejections.`;
  }
  function renderProvenance() {
    const v = audit.main.verification,
      m = audit.main.manifest,
      flagged = v.attempts - v.converged;
    $("hero-fits").textContent = number(v.attempts);
    $("hero-cells").textContent = number(m.cells.length);
    $("hero-reps").textContent = number(m.reps_per_stage_cell);
    $("artifact-record").innerHTML =
      `<p><strong>Primary study complete:</strong> ${number(v.attempts)} attempted fits; ${number(v.completed)} completed; ${number(v.valid)} finite contrast/SE estimates; ${number(v.proper)} proper; ${number(v.converged)} converged. ${flagged ? `${number(flagged)} nonconverged runs were retained, with sensitivity results included.` : "No convergence flags."}</p><p><strong>Fisher benchmark:</strong> ${audit.fisher.rows.length} condition/stage rows and ${audit.fisher.validation.length} independent calculation checks.</p><p><strong>Bootstrap follow-up:</strong> ${audit.bootstrap.complete ? `complete; ${number(audit.bootstrap.manifest.outer_reps * audit.bootstrap.rows.length)} outer surveys and ${number(audit.bootstrap.rows.reduce((n, r) => n + r.bootstrap_attempts, 0))} attempted inner refits.` : "not complete in this artifact; partial results withheld."}</p><p>Schema: <code>${esc(audit.schema)}</code><br>Primary seed: <code>${m.seed}</code><br>${Object.keys(audit.source_sha256).length} source and artifact hashes included in every export. Python ${esc(m.python)} · NumPy ${esc(m.numpy)}.</p>`;
    if (audit.bootstrap_validation)
      $("artifact-record").insertAdjacentHTML(
        "beforeend",
        `<p><strong>Independent null validation:</strong> ${audit.bootstrap_validation.complete ? `${number(audit.bootstrap_validation.manifest.outer_reps)} fresh outer surveys; percentile procedure, fixed negative-pair null. Original and follow-up studies remain separate.` : "prospective protocol recorded; completed results are not available in this artifact."}</p>`,
      );
  }
  function renderValidation() {
    const validation = audit.bootstrap_validation;
    $("null-validation").hidden = !validation;
    if (!validation) return;
    const m = validation.manifest;
    $("validation-design").textContent =
      `A separate negative-pair equality-null study fixes loading at ${m.loading.toFixed(2)} and N at ${number(m.n)}. It uses ${number(m.outer_reps)} fresh surveys and ${number(m.inner_reps)} resampling attempts per survey, comparing the first ${number(m.prefix_resamples)} with all ${number(m.inner_reps)} on those same surveys. The original study is not pooled with this follow-up. Protocol recorded before main computation: ${m.recorded_before_main_computation_utc}.`;
    $("validation-pending").hidden = validation.complete;
    $("validation-results").hidden = !validation.complete;
    if (!validation.complete) return;
    const original = audit.bootstrap.rows.find(
      (r) => r.pair === "negative" && r.stage === "null_validation",
    );
    $("validation-original-rate").textContent = percent(
      original.percentile_reject_rate,
    );
    $("validation-original-context").textContent =
      `${counts(original, "percentile_reject_rate")} null rejections with ${number(original.n_boot)} resamples. This earlier exploratory result prompted the fixed-design follow-up. It is displayed separately and does not contribute surveys to the fresh estimates below.`;
    const fresh = [...validation.rows].sort(
      (a, b) => a.resamples - b.resamples,
    );
    $("validation-rows").innerHTML = fresh
      .map(
        (r) =>
          `<tr class="${r.resamples === m.inner_reps ? "validation-primary" : ""}"><th scope="row"><strong>${number(r.resamples)}</strong><span>${r.resamples === m.inner_reps ? "Primary estimand" : "Paired prefix comparison"}</span></th><td data-label="Null rejections">${counts(r, "false_positive_rate")}</td><td data-label="False-positive rate"><strong>${percent(r.false_positive_rate)}</strong></td><td data-label="95% Wilson MC">${intervalText(r, "false_positive_rate")}</td><td data-label="95% exact MC">${percent(r.exact_cp_low)}–${percent(r.exact_cp_high)}</td></tr>`,
      )
      .join("");
    const paired = validation.paired;
    const pp = (n) =>
      `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n * 100).toFixed(1)}`;
    $("validation-difference").textContent = `${pp(paired.b999_minus_b399)} pp`;
    $("validation-difference-interval").textContent =
      `${number(m.inner_reps)} minus first ${number(m.prefix_resamples)} resamples. Paired 95% Monte Carlo interval: ${pp(paired.paired_bootstrap_low)} to ${pp(paired.paired_bootstrap_high)} percentage points.`;
    if (paired.paired_bootstrap_low < 0 && paired.paired_bootstrap_high > 0)
      $("validation-difference-interval").textContent +=
        " This interval spans zero; the observed change does not resolve whether more resamples raise or lower the rejection rate.";
    $("validation-discordance").textContent =
      `${number(paired.b999_only_reject)} surveys reject only with ${number(m.inner_reps)} resamples; ${number(paired.b399_only_reject)} reject only with the first ${number(m.prefix_resamples)}. Both reject on ${number(paired.both_reject)}; neither rejects on ${number(paired.neither_reject)}. Denominator: the same ${number(paired.attempts)} fresh surveys.`;
    if (paired.b999_only_reject + paired.b399_only_reject === 0)
      $("validation-discordance").textContent +=
        " The empirical paired interval is degenerate because this sample contains no discordant surveys. It does not establish population equality.";
    $("validation-accounting").textContent =
      `All-attempt denominators; failed outer tests count as non-rejections. Computable percentile intervals: ${fresh.map((r) => `${number(r.valid_rate_count)}/${number(r.attempts)} at ${number(r.resamples)} resamples`).join("; ")}. Inner failures retain their attempted positions in the prefix. Fresh seed: ${m.seed}. No numerical outcome was used to extend the fixed stopping rule.`;
    if (validation.verification) {
      const v = validation.verification;
      $("validation-accounting").textContent +=
        ` Outer surveys: ${number(v.outer_valid)} finite estimates, ${number(v.outer_proper)} proper fits, ${number(v.outer_converged)} converged. Inner refits: ${number(v.bootstrap_refits)} attempted, ${number(v.bootstrap_failed)} failed, ${number(v.bootstrap_improper)} improper, ${number(v.bootstrap_nonconverged)} nonconverged. Computability does not establish proper fits, convergence, or calibrated inference.`;
    }
    $("validation-normal").innerHTML = fresh
      .map(
        (r) =>
          `<p><strong>${number(r.resamples)} resamples:</strong> normal intervals with bootstrap SE reject ${percent(r.normal_false_positive_rate)} of null surveys (${counts(r, "normal_false_positive_rate")}; 95% Wilson MC ${intervalText(r, "normal_false_positive_rate")}). This is a secondary companion check; the specified primary estimand uses percentile intervals.</p>`,
      )
      .join("");
  }
  function render() {
    renderControls();
    renderComparison();
    renderMap();
    renderFisher();
    renderBootstrap();
    renderValidation();
    document.body.classList.toggle("no-uncertainty", !state.uncertainty);
  }
  function update(patch) {
    Object.assign(state, patch);
    render();
    writeUrl();
    const r = row("alternative");
    announce(
      `${pairLabel()}, loading ${state.loading}, N ${number(r.n)}. Calibrated alternative rejection ${percent(r.observed_calibrated_rejection_rate)}.`,
    );
  }
  function download(payload, name) {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2) + "\n"], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast(`Downloaded ${name}`);
  }
  function wire() {
    document
      .querySelectorAll("[data-pair]")
      .forEach((b) =>
        b.addEventListener("click", () => update({ pair: b.dataset.pair })),
      );
    document
      .querySelectorAll("[data-loading]")
      .forEach((b) =>
        b.addEventListener("click", () =>
          update({ loading: Number(b.dataset.loading) }),
        ),
      );
    document
      .querySelectorAll("[data-multiplier]")
      .forEach((b) =>
        b.addEventListener("click", () =>
          update({ multiplier: Number(b.dataset.multiplier) }),
        ),
      );
    document.querySelectorAll("[data-metric]").forEach((b, index, tabs) => {
      b.addEventListener("click", () => update({ metric: b.dataset.metric }));
      b.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft")
          next = (index + tabs.length - 1) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next !== undefined) {
          event.preventDefault();
          tabs[next].focus();
          tabs[next].click();
        }
      });
    });
    $("design-map").addEventListener("click", (event) => {
      const button = event.target.closest("[data-map-loading]");
      if (button) {
        const restoreFocus = document.activeElement === button;
        update({
          loading: Number(button.dataset.mapLoading),
          multiplier: Number(button.dataset.mapSize),
        });
        if (restoreFocus)
          document
            .querySelector('.map-cell[aria-pressed="true"]')
            .focus({ preventScroll: true });
        toast(
          "Condition updated. The diagnostic above follows this selection.",
        );
      }
    });
    $("show-uncertainty").addEventListener("change", (event) => {
      state.uncertainty = event.target.checked;
      document.body.classList.toggle("no-uncertainty", !state.uncertainty);
      announce(
        state.uncertainty
          ? "Monte Carlo intervals shown."
          : "Monte Carlo intervals hidden visually; all values remain in exports.",
      );
    });
    $("bootstrap-method").addEventListener("change", (event) =>
      update({ bootstrapMethod: event.target.value }),
    );
    $("copy-condition").addEventListener("click", async () => {
      writeUrl();
      try {
        await navigator.clipboard.writeText(location.href);
        toast("Condition link copied.");
      } catch {
        toast("Copy the address bar URL to share this condition.");
      }
    });
    $("export-condition").addEventListener("click", () =>
      download(
        {
          schema: "sem-model-audit-selection-v1",
          simulation_only: true,
          selection: state,
          alternative: row("alternative"),
          null_validation: row("null_validation"),
          fisher: fisherRow(),
          bootstrap: {
            complete: audit.bootstrap.complete,
            calibration_check: audit.bootstrap.calibration_check || null,
            rows: audit.bootstrap.complete
              ? audit.bootstrap.rows.filter((r) => r.pair === state.pair)
              : [],
          },
          interpretation: audit.interpretation,
          assumptions: audit.assumptions || null,
          reproducibility: audit.reproducibility || null,
          manifests: {
            main: audit.main.manifest,
            fisher: audit.fisher.manifest,
            bootstrap: audit.bootstrap.manifest,
          },
          source_sha256: audit.source_sha256,
        },
        `model-audit-${row("alternative").cell}.json`,
      ),
    );
    $("export-study").addEventListener("click", () =>
      download(audit, "sem-model-audit.json"),
    );
    $("export-validation").addEventListener("click", () => {
      if (!audit.bootstrap_validation?.complete) return;
      download(
        {
          schema: "sem-model-audit-validation-export-v1",
          simulation_only: true,
          independent_followup: audit.bootstrap_validation,
          original_negative_null: audit.bootstrap.rows.find(
            (r) => r.pair === "negative" && r.stage === "null_validation",
          ),
          interpretation: audit.interpretation,
          assumptions: audit.assumptions || null,
          reproducibility: audit.reproducibility || null,
          source_sha256: audit.source_sha256,
        },
        "sem-independent-null-validation.json",
      );
    });
    window.addEventListener("hashchange", () => {
      if (location.hash.includes("pair=")) {
        readUrl();
        render();
      }
    });
  }
  function validate(data) {
    const probability = (value) =>
      Number.isFinite(value) && value >= 0 && value <= 1;
    const validRate = (row, key) => {
      const rate = row[key],
        count = row[key + "_count"],
        denominator = row[key + "_denominator"],
        low = row[key + "_low"],
        high = row[key + "_high"];
      return (
        probability(rate) &&
        Number.isInteger(count) &&
        count >= 0 &&
        count <= denominator &&
        denominator === row.attempts &&
        Math.abs(rate - count / denominator) < 1e-10 &&
        probability(low) &&
        probability(high) &&
        low <= high
      );
    };
    if (
      data.schema !== "sem-model-audit-v1" ||
      data.simulation_only !== true ||
      !data.main?.complete ||
      !data.fisher?.complete
    )
      throw new Error(
        "A completed, verified primary simulation and Fisher artifact are required.",
      );
    const expected = data.main.manifest.cells.length * 2;
    if (data.main.rows.length !== expected)
      throw new Error("The primary summary is incomplete.");
    const identity = new Set();
    for (const r of data.main.rows) {
      const key = `${r.pair}:${r.loading}:${r.multiplier}:${r.stage}`;
      if (identity.has(key))
        throw new Error("A simulation condition is duplicated.");
      identity.add(key);
      if (r.attempts !== data.main.manifest.reps_per_stage_cell)
        throw new Error("An evaluation stream is incomplete.");
      for (const method of methods) {
        if (!validRate(r, method.key))
          throw new Error("An evaluation rate is unavailable.");
      }
      if (
        !probability(r.calibrated_two_sample_bootstrap_low) ||
        !probability(r.calibrated_two_sample_bootstrap_high) ||
        r.calibrated_two_sample_bootstrap_low >
          r.calibrated_two_sample_bootstrap_high
      )
        throw new Error("The calibrated Monte Carlo interval is unavailable.");
    }
    for (const cell of data.main.manifest.cells) {
      for (const stage of ["alternative", "null_validation"]) {
        if (
          !identity.has(
            `${cell.pair}:${cell.loading}:${cell.multiplier}:${stage}`,
          )
        )
          throw new Error("A planned simulation condition is missing.");
      }
    }
    for (const pair of ["positive", "negative"]) {
      for (const loading of [0.5, 0.75, 0.9]) {
        const f = data.fisher.rows.find(
          (r) =>
            r.pair === pair &&
            r.loading === loading &&
            r.stage === "alternative",
        );
        if (
          !f ||
          !Number.isFinite(f.asymptotic_n_for_80pct_power) ||
          !data.fisher.validation.some(
            (r) => r.pair === pair && r.loading === loading,
          )
        )
          throw new Error(
            "A Fisher benchmark or independent validation is missing.",
          );
      }
    }
    if (
      data.bootstrap.complete &&
      (data.bootstrap.rows.length !== 4 ||
        new Set(data.bootstrap.rows.map((r) => `${r.pair}:${r.stage}`)).size !==
          4 ||
        data.bootstrap.rows.some(
          (r) =>
            r.attempts !== data.bootstrap.manifest.outer_reps ||
            !["positive", "negative"].includes(r.pair) ||
            !["alternative", "null_validation"].includes(r.stage) ||
            [
              "oracle_reject_rate",
              "observed_reject_rate",
              "percentile_reject_rate",
              "normal_reject_rate",
              "percentile_coverage_rate",
              "normal_coverage_rate",
              "bootstrap_valid_rate",
            ].some((key) => !validRate(r, key)),
        ))
    )
      throw new Error(
        "The bootstrap artifact claims completion but has incomplete streams.",
      );
    if (data.bootstrap_validation?.complete) {
      const v = data.bootstrap_validation;
      const expectedResamples = [
        v.manifest.prefix_resamples,
        v.manifest.inner_reps,
      ];
      if (
        !data.bootstrap.complete ||
        v.verification?.status !== "passed" ||
        v.verification.outer_reps !== v.manifest.outer_reps ||
        v.verification.no_seed_overlap_with_original_studies !== true ||
        v.rows.length !== 2 ||
        !expectedResamples.every((count) =>
          v.rows.some((r) => r.resamples === count),
        ) ||
        v.rows.some(
          (r) =>
            r.attempts !== v.manifest.outer_reps ||
            r.pair !== v.manifest.pair ||
            r.loading !== v.manifest.loading ||
            r.n !== v.manifest.n ||
            r.stage !== "null_validation" ||
            [
              "false_positive_rate",
              "normal_false_positive_rate",
              "coverage_rate",
              "valid_rate",
            ].some((key) => !validRate(r, key)) ||
            !probability(r.exact_cp_low) ||
            !probability(r.exact_cp_high) ||
            r.exact_cp_low > r.exact_cp_high,
        ) ||
        v.paired?.attempts !== v.manifest.outer_reps ||
        [
          "b999_minus_b399",
          "paired_bootstrap_low",
          "paired_bootstrap_high",
        ].some((key) => !Number.isFinite(v.paired[key])) ||
        [
          "outer_valid",
          "outer_proper",
          "outer_converged",
          "bootstrap_refits",
          "bootstrap_failed",
          "bootstrap_improper",
          "bootstrap_nonconverged",
        ].some(
          (key) =>
            !Number.isInteger(v.verification[key]) || v.verification[key] < 0,
        )
      )
        throw new Error(
          "The independent validation claims completion but is incomplete.",
        );
    }
  }
  async function init() {
    try {
      const response = await fetch("model-audit-data.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      audit = await response.json();
      validate(audit);
      rows = audit.main.rows;
      readUrl();
      render();
      renderProvenance();
      wire();
      document
        .querySelectorAll("button[disabled],input[disabled]")
        .forEach((el) => {
          el.disabled = false;
        });
      document.documentElement.dataset.auditReady = "true";
      announce(
        "Completed primary simulation loaded. All numerical results come from the verified artifact.",
      );
      if (location.hash.includes("pair="))
        $("workbench").scrollIntoView({ block: "start" });
    } catch (error) {
      for (const id of [
        "hero-fits",
        "hero-cells",
        "hero-reps",
        "fit-rate",
        "fit-interval",
        "reversal-rate",
        "reversal-interval",
        "variance-inflation",
        "advertised-n",
        "fisher-n",
        "fisher-power-base",
        "fisher-power-double",
      ])
        $(id).textContent = "—";
      for (const id of [
        "case-reading",
        "design-map",
        "fit-accounting",
        "fisher-validation",
      ])
        $(id).textContent = "";
      $("case-label").textContent = "Verified study results unavailable";
      $("fisher-reading").textContent =
        "The information benchmark could not be verified in this view.";
      $("advertised-bar").style.width = "0";
      $("fisher-bar").style.width = "0";
      $("bootstrap-results").hidden = true;
      $("bootstrap-unavailable").hidden = false;
      $("null-validation").hidden = true;
      document.querySelectorAll("button,input,select").forEach((el) => {
        el.disabled = true;
      });
      $("load-error").hidden = false;
      $("load-error").textContent =
        "The verified study artifact could not be loaded. Numerical results and controls are unavailable; no partial simulation has been substituted. Reload the page or inspect the source repository.";
      $("comparison-rows").innerHTML =
        '<p class="loading-message">Completed results unavailable.</p>';
      $("artifact-record").textContent =
        "The artifact could not be verified in this view.";
      document.documentElement.dataset.auditError = "true";
      announce("Study data unavailable. Numerical comparisons are disabled.");
      console.error(error);
    }
  }
  init();
})();
