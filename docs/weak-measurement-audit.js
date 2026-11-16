/* Render only the separate, verified weak-measurement artifact. No fitting occurs here. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const n = (x) => Number(x).toLocaleString("en-US");
  const pct = (x, digits = 1) =>
    x === null ? "Not defined" : `${(x * 100).toFixed(digits)}%`;
  const pp = (x) =>
    `${x < 0 ? "−" : x > 0 ? "+" : ""}${Math.abs(x * 100).toFixed(1)}`;
  const methods = {
    naive: ["Naive two-step", "Plug-in standard error"],
    percentile: [
      "Percentile bootstrap",
      "Primary procedure · whole-row resampling",
    ],
    normal: ["Normal bootstrap", "Bootstrap standard error · normal interval"],
  };
  let audit, artifactHash;
  const state = { denominator: "all", pair: "percentile" };
  function readState() {
    const params = new URLSearchParams(location.hash.slice(1));
    state.denominator =
      params.get("denominator") === "computable" ? "computable" : "all";
    state.pair = params.get("pair") === "normal" ? "normal" : "percentile";
  }
  function saveState() {
    history.replaceState(
      null,
      "",
      `#denominator=${state.denominator}&pair=${state.pair}`,
    );
  }
  function validate(data) {
    const probability = (x) => Number.isFinite(x) && x >= 0 && x <= 1;
    const count = (x) => Number.isInteger(x) && x >= 0;
    if (
      data.schema !== "sem-weak-measurement-audit-v1" ||
      data.simulation_only !== true ||
      typeof data.complete !== "boolean" ||
      !data.manifest ||
      !data.source_sha256
    )
      throw Error("Invalid study identity");
    const m = data.manifest;
    if (
      m.pair !== "positive" ||
      m.stage !== "null_validation" ||
      m.loading !== 0.5 ||
      m.n !== 385 ||
      m.outer_reps !== 500 ||
      m.inner_reps !== 399
    )
      throw Error("Unexpected study condition");
    if (!data.complete) {
      if (data.rows.length || data.paired || data.accounting)
        throw Error("Partial outcomes must not be published");
      return;
    }
    if (
      data.verification?.status !== "passed" ||
      data.study_verification?.no_overlap_with_previous_studies_or_pilot !==
        true ||
      data.rows.length !== 6
    )
      throw Error("Study verification incomplete");
    for (const method of Object.keys(methods))
      for (const conditioning of ["all_attempts", "computable_only"]) {
        const rows = data.rows.filter(
          (r) => r.method === method && r.conditioning === conditioning,
        );
        if (rows.length !== 1) throw Error("Missing or duplicate result");
        const r = rows[0],
          d = r.rejection_rate_denominator,
          c = r.rejection_rate_count;
        if (
          !count(d) ||
          !count(c) ||
          c > d ||
          d > m.outer_reps ||
          (conditioning === "all_attempts" && d !== m.outer_reps)
        )
          throw Error("Invalid denominator");
        if (d === 0) {
          if (r.rejection_rate !== null)
            throw Error("Undefined rate has a value");
          continue;
        }
        if (
          !probability(r.rejection_rate) ||
          Math.abs(r.rejection_rate - c / d) > 1e-10 ||
          !probability(r.exact_cp_low) ||
          !probability(r.exact_cp_high) ||
          r.exact_cp_low > r.exact_cp_high
        )
          throw Error("Invalid rate or interval");
      }
    const a = data.accounting;
    for (const key of [
      "outer_attempts",
      "inner_planned",
      "inner_attempted",
      "inner_computable",
      "invalid_intervals",
      "observed_completed",
      "observed_valid",
      "observed_proper",
      "observed_converged",
      "bootstrap_valid",
      "bootstrap_failed",
      "bootstrap_improper",
      "bootstrap_nonconverged",
    ])
      if (!count(a[key])) throw Error("Missing fit accounting");
    if (
      a.outer_attempts !== m.outer_reps ||
      a.inner_planned !== m.outer_reps * m.inner_reps ||
      a.inner_attempted > a.inner_planned ||
      a.inner_computable + a.bootstrap_failed !== a.inner_attempted ||
      a.bootstrap_valid + a.invalid_intervals !== a.outer_attempts ||
      a.bootstrap_improper > a.inner_computable ||
      a.bootstrap_nonconverged > a.inner_computable ||
      [
        "observed_completed",
        "observed_valid",
        "observed_proper",
        "observed_converged",
        "bootstrap_valid",
      ].some((key) => a[key] > a.outer_attempts)
    )
      throw Error("Inconsistent accounting");
    const result = (method, conditioning) =>
      data.rows.find(
        (r) => r.method === method && r.conditioning === conditioning,
      );
    for (const method of Object.keys(methods)) {
      const all = result(method, "all_attempts"),
        conditional = result(method, "computable_only");
      if (
        conditional.rejection_rate_denominator !==
          (method === "naive" ? a.observed_valid : a.bootstrap_valid) ||
        conditional.rejection_rate_count !== all.rejection_rate_count
      )
        throw Error("Conditional denominator contradicts fit accounting");
    }
    for (const method of ["percentile", "normal"]) {
      const p = data.paired[method];
      if (
        p.attempts !== m.outer_reps ||
        [
          "neither_reject",
          "bootstrap_only_reject",
          "naive_only_reject",
          "both_reject",
        ].some((key) => !count(p[key])) ||
        p.neither_reject +
          p.bootstrap_only_reject +
          p.naive_only_reject +
          p.both_reject !==
          p.attempts ||
        [
          "bootstrap_minus_naive",
          "paired_bootstrap_low",
          "paired_bootstrap_high",
        ].some((key) => !Number.isFinite(p[key]) || Math.abs(p[key]) > 1) ||
        p.paired_bootstrap_low > p.paired_bootstrap_high ||
        p.bootstrap_only_reject + p.both_reject !==
          result(method, "all_attempts").rejection_rate_count ||
        p.naive_only_reject + p.both_reject !==
          result("naive", "all_attempts").rejection_rate_count ||
        Math.abs(
          p.bootstrap_minus_naive -
            (p.bootstrap_only_reject - p.naive_only_reject) / p.attempts,
        ) > 1e-10
      )
        throw Error("Invalid paired comparison");
    }
  }
  function row(method, conditioning) {
    return audit.rows.find(
      (r) => r.method === method && r.conditioning === conditioning,
    );
  }
  function renderRates() {
    const conditional = state.denominator === "computable";
    $("denominator").value = state.denominator;
    const a = audit.accounting,
      m = audit.manifest;
    $("denominator-note").textContent =
      `${conditional ? "Selected denominator: computable tests only. This conditions on a successful numerical test and can conceal failures if read alone." : "Primary denominator: every attempted survey. Failed or noncomputable tests count as non-rejections; a lower rate can therefore reflect failures as well as uncertainty estimation."} Naive tests: ${n(a.observed_valid)} computable / ${n(m.outer_reps)} attempted; bootstrap intervals: ${n(a.bootstrap_valid)} computable / ${n(m.outer_reps)} attempted. Improper and nonconverged computable fits remain included.`;
    if (a.observed_valid === m.outer_reps && a.bootstrap_valid === m.outer_reps)
      $("denominator-note").textContent +=
        " All tests were computable in this run, so both denominator views give the same rates.";
    const scale = Math.min(
      1,
      Math.max(
        0.1,
        Math.ceil(
          Math.max(...audit.rows.map((r) => r.exact_cp_high ?? 0)) * 10,
        ) / 10,
      ),
    );
    $("rate-rows").innerHTML = Object.entries(methods)
      .map(([key, label]) => {
        const r = row(key, conditional ? "computable_only" : "all_attempts");
        const chart = r.rejection_rate_denominator
          ? `<div class="rate-chart" aria-hidden="true" style="--point:${(r.rejection_rate / scale) * 100}%;--low:${(r.exact_cp_low / scale) * 100}%;--high:${(r.exact_cp_high / scale) * 100}%;--target:${(0.05 / scale) * 100}%"><i class="target"></i><i class="interval"></i><i class="dot"></i><span class="axis-start">0%</span><span class="axis-end">${pct(scale, 0)}</span></div>`
          : '<div class="small">No computable tests; a rate cannot be estimated.</div>';
        return `<div class="rate-row" data-method="${key}"><div><strong class="method-name">${label[0]}</strong><span class="method-subtitle">${label[1]}</span></div>${chart}<div class="rate-numeric"><strong class="frequency">${pct(r.rejection_rate)}</strong><span class="rate-count">${n(r.rejection_rate_count)} / ${n(r.rejection_rate_denominator)} reject</span><span class="rate-interval">${r.rejection_rate_denominator ? `${pct(r.exact_cp_low, 2)}–${pct(r.exact_cp_high, 2)} exact MC` : "Exact MC not defined"}</span></div></div>`;
      })
      .join("");
    const primary = row("percentile", "all_attempts");
    $("rate-reading").textContent =
      `The specified primary estimate is the percentile procedure over all ${n(m.outer_reps)} attempted surveys: ${pct(primary.rejection_rate)} reject the equality null. ${primary.exact_cp_low <= 0.05 && primary.exact_cp_high >= 0.05 ? "Its exact Monte Carlo interval includes the nominal 5% rate." : "Its pointwise exact Monte Carlo interval excludes the nominal 5% rate in this condition."} The conditional view changes the denominator, not the fitted procedure.`;
    if (a.observed_valid === m.outer_reps && a.bootstrap_valid === m.outer_reps)
      $("rate-reading").textContent +=
        " All tests were computable, so this rejection frequency is not explained by failed-test denominator loss.";
    $("fit-reading").hidden = a.bootstrap_nonconverged === 0;
    if (a.bootstrap_nonconverged > 0)
      $("fit-reading").textContent =
        `At the same time, ${n(a.bootstrap_nonconverged)} / ${n(a.inner_computable)} computable inner refits (${pct(a.bootstrap_nonconverged / a.inner_computable)}) were flagged as nonconverged. ${n(a.outer_with_bootstrap_nonconverged.rate_count)} / ${n(m.outer_reps)} bootstrap distributions contained nonconverged refits; ${n(a.outer_with_bootstrap_improper.rate_count)} / ${n(m.outer_reps)} contained improper refits. Flagged finite fits were retained. A near-nominal rejection frequency does not resolve those numerical flags.`;
  }
  function renderPaired() {
    $("pair").value = state.pair;
    const p = audit.paired[state.pair],
      name = methods[state.pair][0];
    $("pair-definition").textContent =
      `${name} minus naive two-step, on the same ${n(p.attempts)} fresh surveys. This comparison always uses all attempts, including noncomputable tests as non-rejections; it does not change with the rate-denominator selector.`;
    $("paired-difference").textContent = `${pp(p.bootstrap_minus_naive)} pp`;
    $("paired-interval").textContent =
      `Paired 95% Monte Carlo interval: ${pp(p.paired_bootstrap_low)} to ${pp(p.paired_bootstrap_high)} percentage points.`;
    $("decision-grid").innerHTML = [
      [p.both_reject, "Both reject"],
      [p.bootstrap_only_reject, `${name} only`],
      [p.naive_only_reject, "Naive only"],
      [p.neither_reject, "Neither rejects"],
    ]
      .map(
        ([value, label]) =>
          `<div><strong>${n(value)}</strong><span>${label}</span></div>`,
      )
      .join("");
    $("paired-reading").textContent =
      p.bootstrap_only_reject + p.naive_only_reject === 0
        ? "No discordant decisions occurred in this sample. A degenerate empirical paired interval does not establish population equality."
        : `${p.paired_bootstrap_low <= 0 && p.paired_bootstrap_high >= 0 ? "The paired interval includes zero; the sign of the rate difference is unresolved at this Monte Carlo precision." : "The paired interval describes a within-condition difference in rejection frequency."} A reduction in false positives alone does not establish nominal calibration, equivalence, or alternative power. The comparison uses no surveys from the separate stronger-measurement studies.`;
  }
  function renderAccounting() {
    const a = audit.accounting;
    const groups = [
      [
        "Outer surveys",
        [
          ["Attempted", "outer_attempts"],
          ["Fit completed", "observed_completed"],
          ["Estimate and naive SE computable", "observed_valid"],
          ["Proper fit", "observed_proper"],
          ["Converged fit", "observed_converged"],
          ["Bootstrap intervals computable", "bootstrap_valid"],
          ["Bootstrap intervals not computable", "invalid_intervals"],
        ],
      ],
      [
        "Inner resamples",
        [
          ["Planned positions", "inner_planned"],
          ["Attempted refits", "inner_attempted"],
          ["Computable refits", "inner_computable"],
          ["Failed refits", "bootstrap_failed"],
          ["Improper computable refits", "bootstrap_improper"],
          ["Nonconverged computable refits", "bootstrap_nonconverged"],
        ],
      ],
    ];
    $("accounting-grid").innerHTML = groups
      .map(
        ([title, fields]) =>
          `<div><h4>${title}</h4>${fields.map(([label, key]) => `<div class="accounting-row"><span>${label}</span><strong>${n(a[key])}</strong></div>`).join("")}</div>`,
      )
      .join("");
    $("accounting-note").textContent =
      `Resampling runs only when the outer estimate and naive standard error are finite and the standard error is positive. A bootstrap interval needs at least 95% computable refits. Improper and nonconverged flags can overlap; flagged computable fits are retained. Inner counts are nested within surveys and are descriptive, not independent binomial trials. ${[
        "failed",
        "improper",
        "nonconverged",
      ]
        .map((flag) => {
          const r = a["outer_with_bootstrap_" + flag];
          return `${n(r.rate_count)} / ${n(r.rate_denominator)} surveys contain at least one ${flag} refit`;
        })
        .join("; ")}.`;
  }
  function render() {
    renderRates();
    renderPaired();
  }
  function download() {
    const payload = {
      schema: "sem-weak-measurement-review-v1",
      simulation_only: true,
      artifact: {
        url: new URL("weak-measurement-audit-data.json", location.href).href,
        sha256: artifactHash,
      },
      view: { ...state },
      study: audit,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], {
        type: "application/json",
      }),
      link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "weak-measurement-review.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    $("export-status").textContent =
      "Exported exact study artifact, source hashes, and current comparison state.";
  }
  async function init() {
    try {
      const response = await fetch("weak-measurement-audit-data.json");
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      const bytes = await response.arrayBuffer();
      audit = JSON.parse(new TextDecoder().decode(bytes));
      validate(audit);
      artifactHash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      )
        .map((x) => x.toString(16).padStart(2, "0"))
        .join("");
      const m = audit.manifest;
      $("design").innerHTML = [
        [m.loading.toFixed(2), "Indicator loading λ"],
        [n(m.n), "Observations per survey"],
        [
          n(m.outer_reps),
          audit.complete ? "Completed fresh surveys" : "Planned fresh surveys",
        ],
        [n(m.inner_reps), "Resampling attempts per eligible survey"],
      ]
        .map(
          ([value, label]) =>
            `<div><strong>${value}</strong><span>${label}</span></div>`,
        )
        .join("");
      $("design").hidden = false;
      $("design-description").textContent =
        `Positive-pair equality null: β₁ = β₂ = ${m.null_coefficients[0]}. ${m.indicators_per_factor} indicators per latent variable; predictor correlation ${m.predictor_correlation}. This is a null experiment; it does not estimate power against an alternative.`;
      $("protocol-record").textContent =
        `Protocol recorded before main computation: ${m.recorded_before_main_computation_utc}. Seed ${m.seed}. Fixed stopping rule and estimator are included in the downloadable protocol. ${audit.complete ? "Saved-draw reproduction checks passed; this does not establish estimator usability or calibration. The new study and timing pilot use separate random streams from earlier studies." : "No completed outcome estimates are published in this artifact."}`;
      $("artifact-record").textContent = `Artifact SHA-256: ${artifactHash}`;
      $("downloads").innerHTML =
        '<a href="weak-measurement-downloads/protocol.json">Recorded protocol · JSON ↓</a>';
      $("downloads").hidden = false;
      $("pending").hidden = audit.complete;
      if (!audit.complete) {
        $("study-status").textContent = "Protocol recorded · results pending";
        document.documentElement.dataset.auditPending = "true";
        return;
      }
      readState();
      $("pair").innerHTML =
        '<option value="percentile">Percentile minus naive</option><option value="normal">Normal minus naive</option>';
      render();
      renderAccounting();
      $("findings").hidden = false;
      $("study-status").textContent = "Complete · reproduction verified";
      $("downloads").insertAdjacentHTML(
        "beforeend",
        '<a href="weak-measurement-methods.md">Methods, checks & limitations ↗</a><a href="weak-measurement-downloads/summary.csv">All six result rows · CSV ↓</a><a href="weak-measurement-downloads/replication.zip">Separate replication bundle · ZIP ↓</a><a href="weak-measurement-downloads/replication.zip.sha256">Bundle checksum · SHA-256 ↓</a>',
      );
      $("export").disabled = false;
      $("export").addEventListener("click", download);
      for (const key of ["denominator", "pair"])
        $(key).addEventListener("change", () => {
          state[key] = $(key).value;
          render();
          saveState();
        });
      window.addEventListener("hashchange", () => {
        readState();
        render();
      });
      document.documentElement.dataset.auditReady = "true";
    } catch (error) {
      for (const id of ["design", "pending", "findings", "downloads"])
        $(id).hidden = true;
      $("design-description").textContent = "";
      $("artifact-record").textContent = "";
      $("protocol-record").textContent =
        "A verified protocol and complete artifact are required to display study findings.";
      $("failure").hidden = false;
      $("study-status").textContent = "Artifact unavailable";
      document
        .querySelectorAll("button,select")
        .forEach((el) => (el.disabled = true));
      document.documentElement.dataset.auditError = "true";
      console.error(error);
    }
  }
  init();
})();
