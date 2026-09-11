#!/usr/bin/env python3
"""Generate API documentation for ehs_risk_sem."""

import importlib
import inspect
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
PKG_NAME = "ehs_risk_sem"
PKG_PATH = REPO_ROOT / PKG_NAME
DOCS_DIR = REPO_ROOT / "docs"
API_DIR = DOCS_DIR / "api"
SIMS_DIR = REPO_ROOT / "simulations"
INDEX_HTML = DOCS_DIR / "index.html"

# Add repo to path so we can import the package
sys.path.insert(0, str(REPO_ROOT))

# CSS template from docs/index.html (copied)
CSS_TEMPLATE = """<style>
/* ============ tokens ============ */
:root {
 color-scheme:light;
 --paper:#F7F7F3; --surface:#FFFFFF; --surface-2:#EFEFE9;
 --ink:#101311; --ink-2:#3A403C; --muted:#666D68;
 --rule:#101311; --rule-soft:#D6D7CE;
 --accent:#1E40AF; --accent-2:#17307F; --accent-ink:#FFFFFF; --accent-wash:#E2E8FB;
 --warn:#8A5A00; --warn-wash:#FBF0DC;
 --grid-dot:rgba(16,19,17,.10); --shadow:rgba(16,19,17,.08);
 --font-sans:"Archivo",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
 --font-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 --maxw:1180px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
 color-scheme:dark;
 --paper:#0A0C10; --surface:#12151C; --surface-2:#171B24;
 --ink:#EAECF2; --ink-2:#BFC4CF; --muted:#838A99;
 --rule:#EAECF2; --rule-soft:#262B36;
 --accent:#7C9EFF; --accent-2:#5B7FE8; --accent-ink:#06101F; --accent-wash:#151C33;
 --warn:#E3B25F; --warn-wash:#241C0C;
 --grid-dot:rgba(234,236,242,.09); --shadow:rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"] {
 color-scheme:dark;
 --paper:#0A0C10; --surface:#12151C; --surface-2:#171B24;
 --ink:#EAECF2; --ink-2:#BFC4CF; --muted:#838A99;
 --rule:#EAECF2; --rule-soft:#262B36;
 --accent:#7C9EFF; --accent-2:#5B7FE8; --accent-ink:#06101F; --accent-wash:#151C33;
 --warn:#E3B25F; --warn-wash:#241C0C;
 --grid-dot:rgba(234,236,242,.09); --shadow:rgba(0,0,0,.4);
}
:root[data-theme="light"]{color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font-sans);font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
h1,h2,h3{margin:0;line-height:1.06;letter-spacing:-.01em}
p{margin:0 0 1em}
a{color:var(--accent)}
img,svg{max-width:100%}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 30px}
.label{font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:600}
.mono{font-family:var(--font-mono);font-variant-numeric:tabular-nums}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
a:focus-visible,button:focus-visible,summary:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,[tabindex]:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
code,.mono{overflow-wrap:anywhere}
@media print{
  .topbar,footer{display:none!important}
  body{background:#fff!important;color:#000!important}
  a{color:#000!important;text-decoration:underline}
  .wrap{max-width:none}
}
.topbar{position:sticky;top:0;z-index:40;background:var(--paper);border-bottom:2px solid var(--rule)}
.topbar-inner{max-width:var(--maxw);margin:0 auto;padding:13px 30px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{display:flex;align-items:center;gap:10px;font-family:var(--font-mono);font-size:.8125rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--ink);text-decoration:none}
.brand-mark{width:14px;height:14px;background:var(--accent);flex:none}
.topnav{display:flex;align-items:center;gap:2px}
.topnav a{text-decoration:none;padding:7px 11px;font-family:var(--font-mono);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-2);transition:color .15s,background .15s}
.topnav a:hover{color:var(--ink);background:var(--surface-2)}
@media(prefers-color-scheme:dark){:root:not([data-theme="light"]) .topnav a{color:var(--muted)}}
.toggle{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;flex:none;background:var(--paper);border:2px solid var(--rule);border-radius:0;color:var(--ink);cursor:pointer}
.toggle:hover{background:var(--surface-2)}
.icon-moon{display:none}
:root[data-theme="dark"] .icon-moon{display:block}
:root[data-theme="dark"] .icon-sun{display:none}
.api-container{display:flex;gap:30px;min-height:calc(100vh - 80px)}
.api-sidebar{flex:0 0 220px;padding:30px 0;border-right:2px solid var(--rule);max-height:calc(100vh - 80px);overflow-y:auto}
.api-main{flex:1;padding:30px 0}
.api-nav-item{padding:8px 0;font-family:var(--font-mono);font-size:.875rem;color:var(--muted);text-decoration:none;display:block}
.api-nav-item:hover{color:var(--ink)}
.api-nav-item.active{color:var(--accent);font-weight:600}
.api-symbol{margin-bottom:48px;padding-bottom:30px;border-bottom:2px solid var(--rule-soft)}
.api-symbol:last-child{border-bottom:0}
.api-sig{font-family:var(--font-mono);font-size:.9rem;background:var(--surface);padding:12px 16px;border:2px solid var(--rule-soft);overflow-x:auto;white-space:pre-wrap;word-break:break-word}
.api-doc{margin-top:12px;color:var(--ink-2);line-height:1.6}
.api-doc p{margin-bottom:.8em}
.api-doc ul{margin-bottom:.8em;padding-left:1.5em}
.api-doc li{margin-bottom:.3em}
.api-used-by{margin-top:12px;padding:10px 12px;background:var(--surface-2);font-family:var(--font-mono);font-size:.8rem;color:var(--ink-2)}
.api-used-by strong{color:var(--ink)}
footer{padding:30px;text-align:center;font-size:.875rem;color:var(--muted);border-top:2px solid var(--rule)}
.back-link{display:inline-block;margin-bottom:20px;padding:8px 0;font-family:var(--font-mono);font-size:.875rem;color:var(--accent);text-decoration:none}
.back-link:hover{text-decoration:underline}
</style>
<script>
document.addEventListener('DOMContentLoaded',function(){
  const theme=localStorage.getItem('theme')||'system';
  if(theme==='system'){
    if(window.matchMedia('(prefers-color-scheme:dark)').matches){
      document.documentElement.setAttribute('data-theme','dark');
    }
  }else{
    document.documentElement.setAttribute('data-theme',theme);
  }
});
</script>"""


def escape_html(text):
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def parse_docstring(doc):
    """Parse docstring into paragraphs and lists.

    Return (description, lists) where description is joined paragraphs
    and lists is list of (title, items) tuples.
    """
    if not doc:
        return "", []

    lines = doc.split('\n')
    # Remove leading/trailing empty lines and common indentation
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return "", []

    # Find common indent
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent == float('inf'):
        min_indent = 0

    # Remove common indent
    lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]

    paragraphs = []
    lists = []
    current_para = []
    current_list = None
    current_list_items = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
            if current_list_items:
                lists.append((current_list, current_list_items))
                current_list = None
                current_list_items = []
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if current_para:
                paragraphs.append('\n'.join(current_para))
                current_para = []
            item = stripped[2:].strip()
            current_list_items.append(item)
        else:
            if current_list_items:
                lists.append((current_list, current_list_items))
                current_list = None
                current_list_items = []
            current_para.append(stripped)

    if current_para:
        paragraphs.append('\n'.join(current_para))
    if current_list_items:
        lists.append((current_list, current_list_items))

    return paragraphs, lists


def format_docstring_html(doc):
    """Format docstring as HTML."""
    if not doc:
        return ""

    paragraphs, lists = parse_docstring(doc)

    html = []
    for para in paragraphs:
        html.append(f"<p>{escape_html(para)}</p>")

    for title, items in lists:
        html.append("<ul>")
        for item in items:
            html.append(f"<li>{escape_html(item)}</li>")
        html.append("</ul>")

    return '\n'.join(html)


def get_signature(obj):
    """Get signature string for function/class."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "()"


def find_used_by(module_name, symbol_name):
    """Find which simulation files import this symbol."""
    used_by = []
    if not SIMS_DIR.exists():
        return used_by

    for sim_file in sorted(SIMS_DIR.glob('*.py')):
        if sim_file.name.startswith('run_') or sim_file.name == '__init__.py':
            continue
        try:
            with open(sim_file) as f:
                content = f.read()
                # Check for direct imports: from ehs_risk_sem.module import symbol
                if f"from {PKG_NAME}.{module_name} import" in content:
                    if symbol_name in content:
                        used_by.append(sim_file.name)
                # Check for module.symbol usage: ehs_risk_sem.module.symbol
                if f"{PKG_NAME}.{module_name}.{symbol_name}" in content:
                    if sim_file.name not in used_by:
                        used_by.append(sim_file.name)
        except Exception:
            pass

    return used_by


def get_public_symbols(module):
    """Get public classes and functions from module."""
    symbols = {}

    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue

        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            symbols[name] = obj
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            symbols[name] = obj

    return symbols


def generate_module_page(module_name):
    """Generate API page for a module."""
    try:
        mod = importlib.import_module(f"{PKG_NAME}.{module_name}")
    except ImportError as e:
        print(f"Could not import {PKG_NAME}.{module_name}: {e}", file=sys.stderr)
        return None

    symbols = get_public_symbols(mod)

    # Build HTML content
    html_parts = []

    for symbol_name in sorted(symbols.keys()):
        obj = symbols[symbol_name]
        sig = get_signature(obj)
        doc = inspect.getdoc(obj)
        used_by = find_used_by(module_name, symbol_name)

        html_parts.append('<div class="api-symbol">')
        html_parts.append(f'<h3>{symbol_name}</h3>')
        html_parts.append(f'<div class="api-sig">{symbol_name}{escape_html(sig)}</div>')

        if doc:
            html_parts.append(f'<div class="api-doc">{format_docstring_html(doc)}</div>')

        if used_by:
            html_parts.append(
                f'<div class="api-used-by"><strong>Used by:</strong> {", ".join(used_by)}</div>'
            )

        html_parts.append('</div>')

    return {
        'module': module_name,
        'symbols': symbols,
        'content': '\n'.join(html_parts),
    }


def get_all_modules():
    """Get list of all public modules in package."""
    modules = []
    for item in sorted(PKG_PATH.glob('*.py')):
        if item.name.startswith('_'):
            continue
        if item.name == '__init__.py':
            continue
        mod_name = item.stem
        modules.append(mod_name)
    return modules


def generate_index_page(all_modules, all_pages):
    """Generate index page with navigation."""
    nav_html = []
    for mod_name in all_modules:
        nav_html.append(
            f'<a href="{mod_name}.html" class="api-nav-item">{mod_name}</a>'
        )

    return '\n'.join(nav_html)


def generate_html_page(title, description, canonical, nav_html, content_html, is_index=False):
    """Generate complete HTML page."""
    back_link = '<a href="../index.html" class="back-link">Back to ehs-risk-sem</a>' if not is_index else ''

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{escape_html(title)}</title>
<meta name="description" content="{escape_html(description)}" />
<link rel="canonical" href="{escape_html(canonical)}" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" />
{CSS_TEMPLATE}
</head>
<body>
<header class="topbar">
  <div class="topbar-inner">
    <a href="../index.html" class="brand"><span class="brand-mark"></span> ehs-risk-sem</a>
    <nav class="topnav">
      <a href="../index.html">Home</a>
      <a href="index.html">API</a>
    </nav>
    <button class="toggle" id="theme-toggle" aria-label="Toggle theme">
      <svg class="icon-sun" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="10" cy="10" r="4" stroke="currentColor" stroke-width="1.5"/>
        <path d="M10 1v3M10 16v3M1 10h3M16 10h3M3.64 3.64l2.12 2.12M14.24 14.24l2.12 2.12M16.36 3.64l-2.12 2.12M5.76 14.24l-2.12 2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <svg class="icon-moon" width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M17.5 10.5c0-1.04-.16-2.04-.47-3 .68.24 1.32.58 1.9 1.01.7.52 1.26 1.18 1.65 1.95-.32.02-.64.04-.96.04-1.5 0-2.95-.31-4.27-.88.05.3.15.59.15.88 0 2.76-2.24 5-5 5s-5-2.24-5-5c0-.29.1-.58.15-.88C3.31 10.69 1.86 11 .36 11c-.32 0-.64-.02-.96-.04.39-.77.95-1.43 1.65-1.95.58-.43 1.22-.77 1.9-1.01-.31.96-.47 1.96-.47 3 0 4.97 4.03 9 9 9 1.04 0 2.04-.16 3-.47-.24.68-.58 1.32-1.01 1.9-.52.7-1.18 1.26-1.95 1.65.02-.32.04-.64.04-.96 0-1.5-.31-2.95-.88-4.27.3.05.59.15.88.15 2.76 0 5-2.24 5-5z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>
  </div>
</header>
<main class="wrap">
  {back_link}
  <div class="api-container">
    <nav class="api-sidebar">
      {nav_html}
    </nav>
    <article class="api-main">
      {content_html}
    </article>
  </div>
</main>
<footer>Generated API reference for ehs-risk-sem</footer>
<script>
document.getElementById('theme-toggle').addEventListener('click',function(){{
  const current=document.documentElement.getAttribute('data-theme')||'system';
  const next=current==='light'?'dark':current==='dark'?'system':'light';
  if(next==='system'){{
    document.documentElement.removeAttribute('data-theme');
    localStorage.removeItem('theme');
  }}else{{
    document.documentElement.setAttribute('data-theme',next);
    localStorage.setItem('theme',next);
  }}
}});
</script>
</body>
</html>"""

    return html


def main():
    """Generate API documentation."""
    API_DIR.mkdir(parents=True, exist_ok=True)

    # Get all modules
    all_modules = get_all_modules()

    if not all_modules:
        print(f"No modules found in {PKG_PATH}", file=sys.stderr)
        return 1

    # Generate pages for each module
    all_pages = {}
    for mod_name in all_modules:
        page_data = generate_module_page(mod_name)
        if page_data:
            all_pages[mod_name] = page_data

    # Generate module pages
    for mod_name in all_modules:
        if mod_name not in all_pages:
            continue

        page_data = all_pages[mod_name]
        nav_html = generate_index_page(all_modules, all_pages)

        # Mark active module in nav
        nav_html = nav_html.replace(
            f'<a href="{mod_name}.html" class="api-nav-item">',
            f'<a href="{mod_name}.html" class="api-nav-item active">'
        )

        content_html = f'<h1>{mod_name}</h1>\n{page_data["content"]}'

        canonical = f"https://priyatham9.github.io/ehs-risk-sem/api/{mod_name}.html"
        description = f"API documentation for {PKG_NAME}.{mod_name}"
        title = f"API: {mod_name} - ehs-risk-sem"

        html = generate_html_page(
            title=title,
            description=description,
            canonical=canonical,
            nav_html=nav_html,
            content_html=content_html,
            is_index=False,
        )

        output_file = API_DIR / f"{mod_name}.html"
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Generated {output_file}")

    # Generate index page
    nav_html = generate_index_page(all_modules, all_pages)

    # Build overview for index
    overview_html = "<h1>API Reference</h1>"
    overview_html += "<p>Core modules in ehs_risk_sem:</p>"
    overview_html += "<ul>"
    for mod_name in all_modules:
        if mod_name in all_pages:
            symbols = all_pages[mod_name]['symbols']
            count = len(symbols)
            overview_html += f'<li><a href="{mod_name}.html">{mod_name}</a> ({count} public symbol{"s" if count != 1 else ""})</li>'
    overview_html += "</ul>"

    canonical = "https://priyatham9.github.io/ehs-risk-sem/api/"
    description = f"Complete API reference for {PKG_NAME} package"
    title = "API Reference - ehs-risk-sem"

    html = generate_html_page(
        title=title,
        description=description,
        canonical=canonical,
        nav_html=nav_html,
        content_html=overview_html,
        is_index=True,
    )

    output_file = API_DIR / "index.html"
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Generated {output_file}")

    # Print summary
    total_symbols = sum(len(all_pages[m]['symbols']) for m in all_pages)
    print(f"\nSummary: {len(all_modules)} modules, {total_symbols} public symbols")

    return 0


if __name__ == '__main__':
    sys.exit(main())
