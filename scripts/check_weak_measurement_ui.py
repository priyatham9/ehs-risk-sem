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


def main():
    import hashlib

    artifact = ROOT / "docs/weak-measurement-audit-data.json"
    raw = artifact.read_bytes()
    data = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    pct = lambda value: "Not defined" if value is None else f"{100 * value:.1f}%"
    if data["complete"]:
        archive = ROOT / "docs/weak-measurement-downloads/replication.zip"
        assert (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            == archive.with_suffix(".zip.sha256").read_text().split()[0]
        )
    with local_site() as base, tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(accept_downloads=True)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        for width in (1440, 390):
            page.set_viewport_size({"width": width, "height": 1000})
            page.goto(base + "weak-measurement-audit.html")
            page.wait_for_selector("html[data-audit-ready],html[data-audit-pending]")
            assert digest in page.locator("#artifact-record").text_content()
            if not data["complete"]:
                assert data["rows"] == [] and data["paired"] is None
                assert page.locator("#pending").is_visible()
                assert not page.locator("#findings").is_visible()
                assert page.locator("#export").is_disabled()
                assert page.locator(".frequency").count() == 0
            else:
                assert not page.locator("#pending").is_visible()
                for denominator, conditioning in (
                    ("all", "all_attempts"),
                    ("computable", "computable_only"),
                ):
                    page.select_option("#denominator", denominator)
                    expected = [
                        next(
                            r
                            for r in data["rows"]
                            if r["method"] == method and r["conditioning"] == conditioning
                        )
                        for method in ("naive", "percentile", "normal")
                    ]
                    assert page.locator(".frequency").all_text_contents() == [
                        pct(r["rejection_rate"]) for r in expected
                    ]
                    assert page.locator(".rate-count").all_text_contents() == [
                        f"{r['rejection_rate_count']:,} / {r['rejection_rate_denominator']:,} reject"
                        for r in expected
                    ]
                    assert page.locator(".rate-interval").all_text_contents() == [
                        f"{100 * r['exact_cp_low']:.2f}%–{100 * r['exact_cp_high']:.2f}% exact MC"
                        if r["rejection_rate_denominator"]
                        else "Exact MC not defined"
                        for r in expected
                    ]
                    for method in ("percentile", "normal"):
                        page.select_option("#pair", method)
                        pair = data["paired"][method]
                        difference = pair["bootstrap_minus_naive"]
                        sign = "−" if difference < 0 else "+" if difference > 0 else ""
                        assert (
                            page.locator("#paired-difference").text_content()
                            == f"{sign}{abs(difference * 100):.1f} pp"
                        )
                        assert (
                            "always uses all attempts"
                            in page.locator("#pair-definition").text_content()
                        )
                        assert page.locator("#decision-grid strong").all_text_contents() == [
                            f"{pair[key]:,}"
                            for key in (
                                "both_reject",
                                "bootstrap_only_reject",
                                "naive_only_reject",
                                "neither_reject",
                            )
                        ]
                        no_overflow(page)
                exported = downloaded_json(page, "#export", tmp)
                assert exported["study"] == data and exported["artifact"]["sha256"] == digest
                assert exported["view"] == {"denominator": "computable", "pair": "normal"}
                page.reload()
                page.wait_for_selector("html[data-audit-ready]")
                assert page.locator("#denominator").input_value() == "computable"
                assert page.locator("#pair").input_value() == "normal"
            no_overflow(page)
        failures = [("http", None), ("schema", {"schema": "invalid"})]
        if data["complete"]:
            corrupt = json.loads(raw)
            corrupt["rows"][0]["rejection_rate"] = None
            failures.append(("missing-rate", corrupt))
            corrupt = json.loads(raw)
            corrupt["verification"]["status"] = "failed"
            failures.append(("failed-verification", corrupt))
            corrupt = json.loads(raw)
            corrupt["paired"]["percentile"]["bootstrap_minus_naive"] += 0.01
            failures.append(("contradictory-paired-difference", corrupt))
            corrupt = json.loads(raw)
            corrupt["accounting"]["observed_valid"] -= 1
            failures.append(("contradictory-computable-denominator", corrupt))
        for name, payload in failures:
            page.route(
                "**/weak-measurement-audit-data.json",
                lambda route, _request, value=payload: route.fulfill(
                    status=503 if value is None else 200,
                    content_type="application/json",
                    body=json.dumps(value),
                ),
            )
            page.goto(base + "weak-measurement-audit.html?failure=" + name)
            page.wait_for_selector("html[data-audit-error]")
            assert page.locator("#failure").is_visible()
            assert not page.locator("#findings").is_visible()
            assert not page.locator("#design").is_visible()
            assert page.locator("#export").is_disabled()
            no_overflow(page)
            page.unroute("**/weak-measurement-audit-data.json")
        assert not errors, errors
        browser.close()
    print(
        "PASS: weak-measurement artifact state, denominators, paired decisions, exact export/hash, desktop/mobile, unavailable state."
    )


if __name__ == "__main__":
    main()
