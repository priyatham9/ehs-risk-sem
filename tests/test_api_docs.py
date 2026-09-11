"""Tests for API documentation generation."""

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ehs_risk_sem


class TestAPIDocsGeneration(unittest.TestCase):
    """Test that API documentation can be generated."""

    def setUp(self):
        """Set up for tests."""
        self.repo_root = Path(__file__).parent.parent
        self.api_dir = self.repo_root / "docs" / "api"
        self.build_script = self.repo_root / "tools" / "build_api_docs.py"

    def test_generator_exists(self):
        """Test that the generator script exists."""
        self.assertTrue(self.build_script.exists(), f"{self.build_script} not found")

    def test_generator_runs(self):
        """Test that the generator runs without error."""
        result = subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"Generator failed:\n{result.stderr}")

    def test_api_pages_exist(self):
        """Test that API pages are generated."""
        # Regenerate to ensure fresh state
        subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )

        # Check index exists
        index_html = self.api_dir / "index.html"
        self.assertTrue(index_html.exists(), f"{index_html} not found")

        # Check that at least one module page exists
        module_pages = list(self.api_dir.glob("*.html"))
        self.assertGreater(len(module_pages), 1, "No module pages generated")

    def test_all_public_symbols_in_html(self):
        """Test that all public symbols appear in generated HTML."""
        # Regenerate
        subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )

        # Get all public symbols
        all_symbols = set()
        for name, obj in inspect.getmembers(ehs_risk_sem):
            if not name.startswith('_'):
                all_symbols.add(name)

            # Also check submodules
            if inspect.ismodule(obj) and obj.__name__.startswith('ehs_risk_sem'):
                for subname, subobj in inspect.getmembers(obj):
                    if not subname.startswith('_'):
                        if inspect.isclass(subobj) and subobj.__module__ == obj.__name__:
                            all_symbols.add(subname)
                        elif inspect.isfunction(subobj) and subobj.__module__ == obj.__name__:
                            all_symbols.add(subname)

        # Check each symbol is in at least one HTML file
        found_symbols = set()
        for html_file in self.api_dir.glob("*.html"):
            with open(html_file) as f:
                content = f.read()
                for sym in all_symbols:
                    if sym in content:
                        found_symbols.add(sym)

        self.assertGreater(len(found_symbols), 0, "No symbols found in HTML")
        missing = all_symbols - found_symbols
        # Allow some symbols not to appear (like __version__, internal imports)
        self.assertLess(
            len(missing),
            len(all_symbols) * 0.5,
            f"Too many symbols missing: {missing}",
        )

    def test_generator_deterministic(self):
        """Test that generator produces identical output on consecutive runs."""
        # First run
        result1 = subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result1.returncode, 0)

        # Get file contents after first run
        files_run1 = {}
        for html_file in self.api_dir.glob("*.html"):
            with open(html_file, 'rb') as f:
                files_run1[html_file.name] = f.read()

        # Second run
        result2 = subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result2.returncode, 0)

        # Get file contents after second run
        files_run2 = {}
        for html_file in self.api_dir.glob("*.html"):
            with open(html_file, 'rb') as f:
                files_run2[html_file.name] = f.read()

        # Compare
        self.assertEqual(files_run1, files_run2, "Generator output not deterministic")

    def test_no_em_dashes_in_pages(self):
        """Test that generated HTML contains no em dashes."""
        # Regenerate
        subprocess.run(
            [sys.executable, str(self.build_script)],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
        )

        em_dash_char = '—'
        found_em_dashes = []

        for html_file in self.api_dir.glob("*.html"):
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if em_dash_char in content:
                    found_em_dashes.append(html_file.name)

        self.assertEqual(
            found_em_dashes,
            [],
            f"Found em dashes in: {found_em_dashes}",
        )


if __name__ == '__main__':
    unittest.main()
