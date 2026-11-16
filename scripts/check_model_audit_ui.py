#!/usr/bin/env python3
"""Exercise the static research interface using only its checked-in artifacts."""

import functools
import json
import tempfile
import threading
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@contextmanager
def local_site():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(ROOT))
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/docs/"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def downloaded_json(page, selector, directory):
    with page.expect_download() as event:
        page.locator(selector).click()
    target = Path(directory) / event.value.suggested_filename
    event.value.save_as(target)
    return json.loads(target.read_text())


def no_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def percent(value):
    return f"{value * 100:.1f}%"


def main():
    import hashlib

    data = json.loads((ROOT / "docs/model-audit-data.json").read_text())
    assert data["main"]["complete"] and data["bootstrap"]["complete"]
    downloads = ROOT / "docs/model-audit-downloads"
    assert (downloads / "model-audit-replication.zip").is_file()
    if data.get("bootstrap_validation", {}).get("complete"):
        assert (downloads / "bootstrap-null-validation-replication.zip").is_file()
    for archive in downloads.glob("*.zip"):
        expected = archive.with_suffix(".zip.sha256").read_text().split()[0]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected, archive.name
    with local_site() as base, tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "model-audit.html")
        page.wait_for_selector("html[data-audit-ready]")
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            # Three contrasting actual cells cover pair, loading, and sample controls.
            for pair, loading, multiplier in (
                ("positive", 0.75, 1),
                ("negative", 0.5, 2),
                ("positive", 0.9, 0.5),
            ):
                for key, value in (
                    ("pair", pair),
                    ("loading", loading),
                    ("multiplier", multiplier),
                ):
                    page.locator(f'[data-{key}="{value}"]').click()
                selected = [
                    r
                    for r in data["main"]["rows"]
                    if (r["pair"], r["loading"], r["multiplier"]) == (pair, loading, multiplier)
                ]
                alt = next(r for r in selected if r["stage"] == "alternative")
                null = next(r for r in selected if r["stage"] == "null_validation")
                keys = (
                    "oracle_rejection_rate",
                    "observed_naive_rejection_rate",
                    "observed_calibrated_rejection_rate",
                )
                assert page.locator(".comparison-row .rate-value").all_text_contents() == [
                    percent(row[key]) for key in keys for row in (alt, null)
                ]
                assert (
                    page.locator(".comparison-row.calibrated .interval-label").first.text_content()
                    == f"95% MC {percent(alt['calibrated_two_sample_bootstrap_low'])}–{percent(alt['calibrated_two_sample_bootstrap_high'])}"
                )
                assert page.locator("#reversal-rate").text_content() == percent(
                    alt["wrong_ranking_given_good_fit"]
                )
                assert page.locator(".map-cell").count() == 9
                for method in ("percentile", "normal"):
                    page.select_option("#bootstrap-method", method)
                    original = [
                        next(
                            r
                            for r in data["bootstrap"]["rows"]
                            if r["pair"] == pair and r["stage"] == stage
                        )
                        for stage in ("null_validation", "alternative")
                    ]
                    assert page.locator("#bootstrap-results .rate-value").all_text_contents() == [
                        percent(row[key])
                        for row in original
                        for key in (
                            "observed_reject_rate",
                            method + "_reject_rate",
                            "oracle_reject_rate",
                        )
                    ]
                exported = downloaded_json(page, "#export-condition", tmp)
                assert exported["alternative"] == alt and exported["null_validation"] == null
                no_overflow(page)
            validation = data.get("bootstrap_validation")
            if validation and validation["complete"]:
                assert not page.locator("#validation-pending").is_visible()
                fresh = sorted(validation["rows"], key=lambda r: r["resamples"])
                actual = page.locator("#validation-rows tr").evaluate_all(
                    "rows => rows.map(r => [...r.querySelectorAll('td')].map(c => c.textContent.trim()))"
                )
                assert actual == [
                    [
                        f"{r['false_positive_rate_count']:,} / {r['false_positive_rate_denominator']:,}",
                        percent(r["false_positive_rate"]),
                        f"{percent(r['false_positive_rate_low'])}–{percent(r['false_positive_rate_high'])}",
                        f"{percent(r['exact_cp_low'])}–{percent(r['exact_cp_high'])}",
                    ]
                    for r in fresh
                ]
                original = next(
                    r
                    for r in data["bootstrap"]["rows"]
                    if r["pair"] == "negative" and r["stage"] == "null_validation"
                )
                assert page.locator("#validation-original-rate").text_content() == percent(
                    original["percentile_reject_rate"]
                )
                delta = validation["paired"]["b999_minus_b399"]
                sign = "+" if delta > 0 else "−" if delta < 0 else ""
                assert (
                    page.locator("#validation-difference").text_content()
                    == f"{sign}{abs(delta * 100):.1f} pp"
                )
                assert validation["verification"]["status"] == "passed"
                assert (
                    downloaded_json(page, "#export-validation", tmp)["independent_followup"]
                    == validation
                )
            elif validation:
                assert page.locator("#validation-pending").is_visible()
                assert not page.locator("#validation-results").is_visible()
                assert page.locator("#validation-rows tr").count() == 0
                assert validation["rows"] == [] and validation["paired"] is None
            no_overflow(page)
        link = page.url
        page.reload()
        page.wait_for_selector("html[data-audit-ready]")
        assert page.url == link and page.locator("#bootstrap-method").input_value() == "normal"
        corrupt = json.loads(json.dumps(data))
        corrupt["bootstrap"]["rows"][0]["percentile_reject_rate"] = None
        late_failure = json.loads(json.dumps(data))
        late_failure["source_sha256"] = None
        for name, payload in (
            ("http", None),
            ("missing-rate", corrupt),
            ("late-provenance", late_failure),
        ):
            page.route(
                "**/model-audit-data.json",
                lambda route, _request, value=payload: route.fulfill(
                    status=503 if value is None else 200,
                    content_type="application/json",
                    body=json.dumps(value),
                ),
            )
            page.goto(base + "model-audit.html?failure=" + name)
            page.wait_for_selector("html[data-audit-error]")
            assert page.locator("#hero-fits").text_content() == "—"
            assert page.locator("#fit-rate").text_content() == "—"
            assert not page.locator("#bootstrap-results").is_visible()
            assert not page.locator("#null-validation").is_visible()
            assert page.locator("button:enabled,input:enabled,select:enabled").count() == 0
            no_overflow(page)
            page.unroute("**/model-audit-data.json")
        assert not errors, errors
        browser.close()
    print(
        "PASS: actual conditions/MC intervals; bootstrap scope; exports/link; fresh-study separation; ZIP hashes; unavailable state; desktop/mobile."
    )


if __name__ == "__main__":
    main()
