#!/usr/bin/env python3
"""Generate the HTML API reference for ehs_risk_sem from its docstrings.

Writes one page per public module plus an index to ``docs/api/``, and a small
``docs/api/search-index.js`` that powers the search box on every page. The
output is deterministic: running the script twice produces identical files.

Usage::

    python tools/build_api_docs.py
"""

import importlib
import inspect
import json
import re
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
PKG_NAME = "ehs_risk_sem"
PKG_PATH = REPO_ROOT / PKG_NAME
DOCS_DIR = REPO_ROOT / "docs"
API_DIR = DOCS_DIR / "api"
SIMS_DIR = REPO_ROOT / "simulations"
SITE = "https://priyatham9.github.io/ehs-risk-sem"


def _load_apply_banner():
    """Find grounded's shared banner (tools/banner.py).

    Looks in $GROUNDED_TOOLS first, then a sibling ../grounded/tools checkout.
    Returns None when neither exists; pages are then written without the
    shared banner (the sidebar, search and theme still work on their own).
    """
    import os
    for cand in (os.environ.get("GROUNDED_TOOLS"), str(REPO_ROOT.parent / "grounded" / "tools")):
        if cand and (Path(cand) / "banner.py").exists():
            sys.path.insert(0, cand)
            try:
                from banner import apply_banner
            finally:
                sys.path.remove(cand)
            return apply_banner
    print("note: grounded tools/banner.py not found; API pages built without the shared banner",
          file=sys.stderr)
    return None


APPLY_BANNER = _load_apply_banner()

# Add repo to path so we can import the package
sys.path.insert(0, str(REPO_ROOT))

# Theme init runs in <head> before first paint, with the same storage keys as the
# rest of the site, so a reader's light/dark choice carries over to these pages.
THEME_INIT = (
    '<script>try{var t=localStorage.getItem("pc-theme")||localStorage.getItem("ehs-ai-theme");'
    'if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}'
    "catch(e){}</script>"
)

# Tokens and type follow docs/index.html (Archivo + IBM Plex Mono, 2px rules, blue accent).
CSS_TEMPLATE = """<style>
/* ============ tokens ============ */
:root {
 color-scheme:light;
 --paper:#F7F7F3; --surface:#FFFFFF; --surface-2:#EFEFE9;
 --ink:#101311; --ink-2:#3A403C; --muted:#5F665F;
 --rule:#101311; --rule-soft:#D6D7CE;
 --accent:#1E40AF; --accent-2:#17307F; --accent-ink:#FFFFFF; --accent-wash:#E2E8FB;
 --code-bg:#F1F2EC; --code-ink:#1A1F1C; --mark:#FFE9A8;
 --font-sans:"Archivo",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
 --font-mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
 --maxw:1240px; --bar:58px; /* shared rs-header banner: 56px row + 2px rule */
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
 color-scheme:dark;
 --paper:#0A0C10; --surface:#12151C; --surface-2:#171B24;
 --ink:#EAECF2; --ink-2:#BFC4CF; --muted:#9097A6;
 --rule:#EAECF2; --rule-soft:#262B36;
 --accent:#7C9EFF; --accent-2:#5B7FE8; --accent-ink:#06101F; --accent-wash:#151C33;
 --code-bg:#0F1218; --code-ink:#DDE2EE; --mark:#4A3B0C;
  }
}
:root[data-theme="dark"] {
 color-scheme:dark;
 --paper:#0A0C10; --surface:#12151C; --surface-2:#171B24;
 --ink:#EAECF2; --ink-2:#BFC4CF; --muted:#9097A6;
 --rule:#EAECF2; --rule-soft:#262B36;
 --accent:#7C9EFF; --accent-2:#5B7FE8; --accent-ink:#06101F; --accent-wash:#151C33;
 --code-bg:#0F1218; --code-ink:#DDE2EE; --mark:#4A3B0C;
}
:root[data-theme="light"]{color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-padding-top:calc(var(--bar) + 16px)}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--font-sans);font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased;overflow-x:clip}
h1,h2,h3,h4{margin:0;line-height:1.1;letter-spacing:-.01em}
p{margin:0 0 1em}
a{color:var(--accent)}
code,pre,kbd{font-family:var(--font-mono);font-size:.875em}
code{overflow-wrap:anywhere;word-break:break-word}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%}
pre code{white-space:inherit}
.lede pre{margin:0 0 1em;padding:10px 12px;background:var(--code-bg);color:var(--code-ink);border:2px solid var(--rule-soft);font-size:.8125rem}
.lede pre code{padding:0;border:0;background:none}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
a:focus-visible,button:focus-visible,summary:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.skip{position:absolute;left:-9999px;top:0;z-index:100;background:var(--accent);color:var(--accent-ink);padding:11px 17px;font-family:var(--font-mono);font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;text-decoration:none}
.skip:focus{left:0}
@media (prefers-reduced-motion: reduce){*{transition:none!important;scroll-behavior:auto!important}}


/* ============ layout ============ */
.layout{max-width:var(--maxw);margin:0 auto;padding:0 24px;display:grid;grid-template-columns:260px minmax(0,1fr);gap:40px}
.side{position:sticky;top:var(--bar);align-self:start;max-height:calc(100vh - var(--bar));overflow-y:auto;padding:24px 4px 40px 0;border-right:2px solid var(--rule);scrollbar-width:thin}
.side > summary{display:none}
.side-k{display:block;margin:22px 0 6px;font-family:var(--font-mono);font-size:.6875rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.side ul{list-style:none;margin:0;padding:0}
.side a{display:block;padding:5px 10px;border-left:3px solid transparent;font-family:var(--font-mono);font-size:.8125rem;color:var(--ink-2);text-decoration:none;overflow-wrap:anywhere}
.side a:hover{color:var(--ink);background:var(--surface-2)}
.side a[aria-current="page"],.side a.is-here{color:var(--accent);border-left-color:var(--accent);font-weight:700;background:var(--accent-wash)}
.side .toc a{font-size:.75rem;padding:3px 10px 3px 16px}
.side .toc .kind{color:var(--muted);font-weight:400}
.main{padding:28px 0 64px;min-width:0}

/* ============ search ============ */
.search{position:relative}
.search label{display:block;margin-bottom:6px;font-family:var(--font-mono);font-size:.6875rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.search input{width:100%;height:42px;padding:0 38px 0 12px;border:2px solid var(--rule);border-radius:0;background:var(--surface);color:var(--ink);font:500 .875rem var(--font-mono)}
.search input::placeholder{color:var(--muted)}
.search kbd{position:absolute;right:10px;top:31px;padding:0 5px;border:1px solid var(--rule-soft);font-size:.6875rem;color:var(--muted);pointer-events:none}
.results{list-style:none;margin:6px 0 0;padding:0;border:2px solid var(--rule);background:var(--surface);max-height:60vh;overflow-y:auto}
.results[hidden]{display:none}
.side .results a{display:block;padding:8px 10px;border-bottom:1px solid var(--rule-soft);border-left:0;font-size:.8125rem;color:var(--ink)}
.side .results a:hover,.side .results a.is-active{background:var(--accent-wash)}
.results b{color:var(--accent)}
.results small{display:block;margin-top:2px;font-family:var(--font-sans);font-size:.75rem;line-height:1.35;color:var(--ink-2)}
.results .none{padding:8px 10px;font-size:.8125rem;color:var(--muted)}
mark{background:var(--mark);color:inherit;padding:0}

/* ============ content ============ */
.crumbs{margin:0 0 14px;font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.crumbs a{color:var(--muted)}
.main h1{font-size:clamp(1.9rem,4.4vw,2.8rem);font-stretch:112%;font-weight:900;overflow-wrap:anywhere}
.main h1 code{font-size:1em}
.lede{margin:14px 0 0;max-width:72ch;color:var(--ink-2)}
.lede p{margin:0 0 .8em}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 8px}
.chip{display:inline-block;padding:4px 9px;border:2px solid var(--rule-soft);font-family:var(--font-mono);font-size:.6875rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);overflow-wrap:anywhere}
.api-symbol{margin-top:40px;padding-top:28px;border-top:2px solid var(--rule)}
.sym-head{display:flex;align-items:baseline;flex-wrap:wrap;gap:6px 12px}
.sym-head h2{font-family:var(--font-mono);font-size:clamp(1.15rem,2.4vw,1.45rem);font-weight:700;overflow-wrap:anywhere}
.kind{font-family:var(--font-mono);font-size:.625rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.anchor{margin-left:auto;font-family:var(--font-mono);font-size:.75rem;color:var(--muted);text-decoration:none}
.anchor:hover{color:var(--accent)}
.api-sig,.api-doc pre{margin:12px 0 0;padding:12px 14px;background:var(--code-bg);color:var(--code-ink);border:2px solid var(--rule-soft);border-left:4px solid var(--accent);font-size:.8125rem;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere;tab-size:4}
.api-sig code,.api-doc pre code{font-size:inherit;padding:0;background:none;border:0}
.api-doc pre{border-left-color:var(--rule-soft);margin-bottom:1em}
.api-doc{margin-top:14px;color:var(--ink-2);max-width:78ch}
.api-doc p{margin-bottom:.85em}
.api-doc ul{margin:0 0 .85em;padding-left:1.3em}
.api-doc li{margin-bottom:.3em}
.api-doc code,.lede code{padding:.05em .3em;background:var(--code-bg);color:var(--code-ink);border:1px solid var(--rule-soft);overflow-wrap:anywhere}
.api-doc h4{margin:1.3em 0 .5em;font-family:var(--font-mono);font-size:.6875rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.api-doc dl{margin:0 0 1em;border-top:1px solid var(--rule-soft)}
.api-doc dt{padding-top:8px;font-family:var(--font-mono);font-size:.875rem;font-weight:700;color:var(--ink);overflow-wrap:anywhere}
.api-doc dd{margin:2px 0 0;padding:0 0 8px 14px;border-bottom:1px solid var(--rule-soft)}
.api-doc dd p{margin-bottom:.4em}
.methods{margin-top:18px;padding-left:14px;border-left:2px solid var(--rule-soft)}
.methods h3{margin-top:18px;font-family:var(--font-mono);font-size:1rem;overflow-wrap:anywhere}
.api-used-by{margin-top:14px;padding:10px 12px;background:var(--surface-2);font-family:var(--font-mono);font-size:.75rem;color:var(--ink-2);overflow-wrap:anywhere}
.api-used-by strong{color:var(--ink)}
.to-top{display:inline-block;margin-top:12px;font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);text-decoration:none}
.to-top:hover{color:var(--accent)}
.mods{list-style:none;margin:26px 0 0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));border-top:2px solid var(--rule);border-left:2px solid var(--rule)}
.mods li{border-right:2px solid var(--rule);border-bottom:2px solid var(--rule);background:var(--surface)}
.mods a{display:flex;flex-direction:column;gap:6px;height:100%;padding:16px 18px;color:var(--ink);text-decoration:none}
.mods a:hover{background:var(--surface-2)}
.mods b{font-family:var(--font-mono);font-size:1rem;color:var(--accent)}
.mods span{font-size:.875rem;line-height:1.45;color:var(--ink-2)}
.mods small{margin-top:auto;font-family:var(--font-mono);font-size:.6875rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
footer{border-top:2px solid var(--rule);padding:26px 24px 40px;text-align:center;font-family:var(--font-mono);font-size:.75rem;color:var(--muted)}
footer a{color:var(--muted)}

/* ============ phone ============ */
@media (max-width: 860px){
  .layout{grid-template-columns:minmax(0,1fr);gap:0;padding:0 16px}
  .side{position:static;max-height:none;overflow:visible;padding:14px 0 0;border-right:0;border-bottom:2px solid var(--rule)}
  .side > summary{display:flex;align-items:center;justify-content:space-between;min-height:44px;padding:0 12px;border:2px solid var(--rule);background:var(--surface);font-family:var(--font-mono);font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;list-style:none}
  .side > summary::-webkit-details-marker{display:none}
  .side > summary::after{content:"+";font-size:1.1rem}
  .side[open] > summary::after{content:"\\2212"}
  .side[open]{padding-bottom:16px}
  .side:not([open]){padding-bottom:14px}
  .side-body{padding-top:12px}
  .search kbd{display:none}
  .main{padding-top:18px}
  .api-sig,.api-doc pre{font-size:.75rem;padding:10px 11px}
}
@media print{
  .side,footer,.to-top,.anchor{display:none!important}
  .layout{display:block}
  body{background:#fff!important;color:#000!important}
}
</style>"""

# Page behaviour: search across every page, phone contents drawer,
# and highlighting the symbol currently in view. Plain JS, no dependencies.
PAGE_JS = """<script>
(function(){
  /* theme: the shared banner's toggle sets data-theme on <html>; the colour tokens follow it */

  /* contents drawer: always open on wide screens, collapsed on phones */
  var side=document.getElementById('side'), wide=matchMedia('(min-width: 861px)');
  function fit(){ side.open=wide.matches; }
  fit(); if(wide.addEventListener) wide.addEventListener('change',fit);
  var here=side.querySelector('.side-body > ul [aria-current="page"]');
  if(here&&wide.matches&&here.offsetTop>side.clientHeight-60) side.scrollTop=here.offsetTop-120;

  /* search across every module, from search-index.js */
  var q=document.getElementById('api-q'), list=document.getElementById('api-results');
  function esc(s){return s.replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function hi(s,t){var i=s.toLowerCase().indexOf(t);return i<0?esc(s):esc(s.slice(0,i))+'<mark>'+esc(s.slice(i,i+t.length))+'</mark>'+esc(s.slice(i+t.length));}
  var active=-1;
  function render(){
    var idx=window.API_INDEX||[], t=q.value.trim().toLowerCase(); active=-1; q.removeAttribute('aria-activedescendant');
    if(!t){list.hidden=true;list.innerHTML='';q.setAttribute('aria-expanded','false');return;}
    var hits=idx.map(function(e){var n=e.n.toLowerCase(),s=(e.s||'').toLowerCase(),m=e.m.toLowerCase(),sc=-1;
      if(n===t)sc=0;else if(n.indexOf(t)===0)sc=1;else if(n.indexOf(t)>0)sc=2;else if(m.indexOf(t)>=0)sc=3;else if(s.indexOf(t)>=0)sc=4;
      return {e:e,sc:sc};}).filter(function(h){return h.sc>=0;}).sort(function(a,b){return a.sc-b.sc||a.e.n.localeCompare(b.e.n);}).slice(0,14);
    list.innerHTML=hits.length?hits.map(function(h,i){var e=h.e;return '<li><a role="option" id="api-r'+i+'" href="'+e.u+'"><b>'+hi(e.n,t)+'</b> <span class="kind">'+esc(e.k)+' \\u00b7 '+esc(e.m)+'</span>'+(e.s?'<small>'+hi(e.s,t)+'</small>':'')+'</a></li>';}).join(''):'<li class="none">No symbol matches \\u201c'+esc(q.value)+'\\u201d</li>';
    list.hidden=false;q.setAttribute('aria-expanded','true');
  }
  function move(d){var a=list.querySelectorAll('a');if(!a.length)return;if(active>=0)a[active].classList.remove('is-active');active=(active+d+a.length)%a.length;a[active].classList.add('is-active');a[active].scrollIntoView({block:'nearest'});q.setAttribute('aria-activedescendant',a[active].id);}
  q.addEventListener('input',render);
  q.addEventListener('keydown',function(e){
    if(e.key==='ArrowDown'){e.preventDefault();move(1);} else if(e.key==='ArrowUp'){e.preventDefault();move(-1);}
    else if(e.key==='Enter'){var a=list.querySelectorAll('a');var go=a[active>=0?active:0];if(go){e.preventDefault();location.href=go.href;}}
    else if(e.key==='Escape'){q.value='';render();}
  });
  document.addEventListener('keydown',function(e){
    var ae=document.activeElement;
    if(e.key==='/'&&ae!==q&&!/input|textarea|select/i.test(ae.tagName)){e.preventDefault();side.open=true;q.focus();}
  });

  /* highlight the symbol in view under "On this page" */
  var toc=[].slice.call(side.querySelectorAll('.toc a'));
  if(toc.length&&'IntersectionObserver' in window){
    var map={};toc.forEach(function(a){map[a.getAttribute('href').slice(1)]=a;});
    var io=new IntersectionObserver(function(es){es.forEach(function(en){if(en.isIntersecting){toc.forEach(function(a){a.classList.remove('is-here');});var a=map[en.target.id];if(a)a.classList.add('is-here');}});},{rootMargin:'-20% 0px -70% 0px'});
    toc.forEach(function(a){var s=document.getElementById(a.getAttribute('href').slice(1));if(s)io.observe(s);});
  }
})();
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


# Filled per run: symbol name -> URL, so :func:`name` references become links.
SYMBOL_URLS = {}


def inline_markup(text):
    """Escape text, then render reStructuredText inline code and cross references."""
    out = escape_html(text)

    def ref(match):
        name = match.group(2)
        url = SYMBOL_URLS.get(name.split(".")[-1])
        code = f"<code>{name}</code>"
        return f'<a href="{url}">{code}</a>' if url else code

    out = re.sub(r":(func|class|meth|mod|attr|data):`~?([^`]+)`", ref, out)
    out = re.sub(r"``([^`]+)``", r"<code>\1</code>", out)
    return out


def _indent(line):
    return len(line) - len(line.lstrip())


def normalize_docstring(doc):
    """Undo stray indentation so section headings sit at column zero.

    A few docstrings underline headings with a dash row indented by one space.
    ``inspect.getdoc`` then dedents by that single space and leaves the rest of
    the body indented; aligning the dash row with its heading and dedenting
    again restores the intended layout.
    """
    lines = doc.expandtabs(4).split("\n")
    for k in range(1, len(lines)):
        s = lines[k].strip()
        if s and set(s) == {"-"} and lines[k - 1].strip():
            lines[k] = " " * _indent(lines[k - 1]) + s
    body = [ln for ln in lines[1:] if ln.strip()]
    if body:
        cut = min(_indent(ln) for ln in body)
        lines = lines[:1] + [ln[cut:] if ln.strip() else "" for ln in lines[1:]]
    return lines


def format_docstring_html(doc):
    """Render a cleaned docstring (from inspect.getdoc) as HTML.

    Handles paragraphs, ``-``/``*`` bullet lists, numpy-style section headings
    (a title underlined with dashes), name/description entries inside those
    sections, and literal blocks (any indented block outside a section), which
    become <pre> so they keep their layout on every screen.
    """
    if not doc:
        return ""
    lines = normalize_docstring(doc)
    html = []
    para = []
    items = []
    section = None

    def flush_para():
        if para:
            html.append(f"<p>{inline_markup(' '.join(para))}</p>")
            para.clear()

    def flush_items():
        if items:
            html.append("<ul>" + "".join(f"<li>{inline_markup(i)}</li>" for i in items) + "</ul>")
            items.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        nxt = lines[i + 1].strip() if i + 1 < n else ""

        # Section heading: "Parameters" underlined by "----------"
        if stripped and nxt and set(nxt) == {"-"} and len(nxt) >= 3 and _indent(line) == 0:
            flush_para()
            flush_items()
            section = stripped
            html.append(f"<h4>{escape_html(stripped)}</h4>")
            i += 2
            continue

        if not stripped:
            flush_para()
            flush_items()
            i += 1
            continue

        # Literal block: an indented run of lines outside a definition section
        if _indent(line) > 0 and (section is None or section.lower().startswith("example")):
            flush_para()
            flush_items()
            block = []
            base = _indent(line)
            while i < n and (not lines[i].strip() or _indent(lines[i]) >= base):
                block.append(lines[i][base:] if lines[i].strip() else "")
                i += 1
            while block and not block[-1]:
                block.pop()
            html.append(f"<pre><code>{escape_html(chr(10).join(block))}</code></pre>")
            continue

        # Inside Parameters/Returns/...: an unindented line names an entry and
        # the indented lines below it describe it
        if section is not None and not section.lower().startswith("example"):
            if _indent(line) == 0:
                flush_para()
                i += 1
                desc = []
                while i < n and (not lines[i].strip() or _indent(lines[i]) > 0):
                    desc.append(lines[i].strip())
                    i += 1
                paras = " ".join(d if d else "\n" for d in desc).split("\n")
                body = "".join(
                    f"<p>{inline_markup(p.strip())}</p>" for p in paras if p.strip()
                )
                html.append(f"<dl><dt>{inline_markup(stripped)}</dt><dd>{body}</dd></dl>")
                continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            items.append(stripped[2:].strip())
            i += 1
            continue

        if items and not para:
            items[-1] += " " + stripped
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    flush_items()
    return "\n".join(html)


def first_sentence(doc):
    """First line of a docstring, for summaries and the search index."""
    if not doc:
        return ""
    line = doc.strip().split("\n")[0].strip()
    return re.sub(r":\w+:`~?([^`]+)`", r"\1", line).replace("``", "")


def split_params(inner):
    """Split a parameter list on top-level commas."""
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def format_signature(name, obj):
    """Signature text, one parameter per line when it would not fit on one."""
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = "()"
    sig = sig.replace("'", "")
    one = f"{name}{sig}"
    if len(one) <= 60:
        return one
    depth = 0
    close = len(sig) - 1
    for k, ch in enumerate(sig):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close = k
                break
    params = split_params(sig[1:close])
    if not params:
        return one
    tail = sig[close + 1:]
    body = ",\n".join("    " + p for p in params)
    return f"{name}(\n{body},\n){tail}"


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
                if f"from {PKG_NAME}.{module_name} import" in content:
                    if symbol_name in content:
                        used_by.append(sim_file.name)
                if f"{PKG_NAME}.{module_name}.{symbol_name}" in content:
                    if sim_file.name not in used_by:
                        used_by.append(sim_file.name)
        except Exception:
            pass

    return used_by


def get_public_symbols(module):
    """Get public classes and functions defined in a module."""
    symbols = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            symbols[name] = obj
        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
            symbols[name] = obj
    return symbols


def public_methods(cls):
    """Public methods and properties defined on the class itself."""
    out = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if inspect.isfunction(member):
            out.append((name, member, "method"))
        elif isinstance(member, property):
            out.append((name, member, "property"))
    return out


def get_all_modules():
    """Get list of all public modules in package."""
    return [p.stem for p in sorted(PKG_PATH.glob('*.py')) if not p.name.startswith('_')]


def generate_module_page(module_name):
    """Import one module and collect what its page needs."""
    try:
        mod = importlib.import_module(f"{PKG_NAME}.{module_name}")
    except ImportError as e:
        print(f"Could not import {PKG_NAME}.{module_name}: {e}", file=sys.stderr)
        return None
    return {
        'module': module_name,
        'symbols': get_public_symbols(mod),
        'doc': inspect.getdoc(mod) or "",
    }


def symbol_kind(obj):
    return "class" if inspect.isclass(obj) else "function"


def render_module_content(page):
    """Symbol sections for a module page."""
    module_name = page['module']
    parts = []
    for symbol_name in sorted(page['symbols']):
        obj = page['symbols'][symbol_name]
        kind = symbol_kind(obj)
        doc = inspect.getdoc(obj)
        used_by = find_used_by(module_name, symbol_name)
        parts.append(f'<section class="api-symbol" id="{symbol_name}">')
        parts.append(
            f'<div class="sym-head"><span class="kind">{kind}</span>'
            f'<h2>{symbol_name}</h2>'
            f'<a class="anchor" href="#{symbol_name}" aria-label="Link to {symbol_name}">#</a></div>'
        )
        sig = format_signature(symbol_name, obj)
        parts.append(f'<pre class="api-sig"><code>{escape_html(sig)}</code></pre>')
        if doc:
            parts.append(f'<div class="api-doc">{format_docstring_html(doc)}</div>')
        if kind == "class":
            methods = public_methods(obj)
            if methods:
                parts.append('<div class="methods">')
                for mname, member, mkind in methods:
                    mdoc = inspect.getdoc(member)
                    parts.append(
                        f'<h3 id="{symbol_name}.{mname}"><span class="kind">{mkind}</span> '
                        f'{symbol_name}.{mname}</h3>'
                    )
                    if mkind == "method":
                        msig = format_signature(mname, member)
                        parts.append(f'<pre class="api-sig"><code>{escape_html(msig)}</code></pre>')
                    if mdoc:
                        parts.append(f'<div class="api-doc">{format_docstring_html(mdoc)}</div>')
                parts.append('</div>')
        if used_by:
            parts.append(
                f'<div class="api-used-by"><strong>Used by:</strong> {", ".join(used_by)}</div>'
            )
        parts.append('<a class="to-top" href="#top">Back to top</a>')
        parts.append('</section>')
    return '\n'.join(parts)


def sidebar_html(all_modules, current=None, symbols=None):
    """Search box, module list and, on module pages, the symbols on this page."""
    mods = []
    for mod_name in all_modules:
        cur = ' aria-current="page"' if mod_name == current else ''
        mods.append(f'<li><a href="{mod_name}.html"{cur}>{mod_name}</a></li>')
    toc = ""
    if symbols:
        rows = []
        for name in sorted(symbols):
            kind = "class" if inspect.isclass(symbols[name]) else "fn"
            rows.append(f'<li><a href="#{name}">{name} <span class="kind">{kind}</span></a></li>')
        toc = (
            '<span class="side-k" id="toc-h">On this page</span>'
            f'<ul class="toc" aria-labelledby="toc-h">{"".join(rows)}</ul>'
        )
    return f"""<details class="side" id="side" open>
    <summary>Modules and contents</summary>
    <nav class="side-body" aria-label="API reference">
    <div class="search" role="search">
      <label for="api-q">Search all modules</label>
      <input id="api-q" type="search" placeholder="e.g. rmsea, bootstrap" autocomplete="off"
        spellcheck="false" role="combobox" aria-autocomplete="list" aria-expanded="false"
        aria-controls="api-results" />
      <kbd aria-hidden="true">/</kbd>
      <ul class="results" id="api-results" role="listbox" aria-label="Search results" hidden></ul>
    </div>
    <span class="side-k" id="mods-h">Modules</span>
    <ul aria-labelledby="mods-h">{''.join(mods)}</ul>
    {toc}
    </nav>
  </details>"""


def generate_html_page(title, description, canonical, side, content_html, is_index=False):
    """Generate a complete HTML page."""
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{escape_html(title)}</title>
<meta name="description" content="{escape_html(description)}" />
<link rel="canonical" href="{escape_html(canonical)}" />
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' fill='%231E40AF'/%3E%3Ctext x='8' y='12' font-family='monospace' font-size='11' font-weight='700' text-anchor='middle' fill='white'%3ES%3C/text%3E%3C/svg%3E" />
<meta name="theme-color" content="#F7F7F3" media="(prefers-color-scheme: light)" />
<meta name="theme-color" content="#0A0C10" media="(prefers-color-scheme: dark)" />
{THEME_INIT}
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" />
{CSS_TEMPLATE}
<script src="search-index.js" defer></script>
</head>
<body id="top">
<a class="skip" href="#main">Skip to content</a>
<div class="layout">
  {side}
  <main class="main" id="main">
{content_html}
  </main>
</div>
<footer>Generated from the package docstrings by <code>tools/build_api_docs.py</code> &middot;
<a href="https://github.com/priyatham9/ehs-risk-sem">Source on GitHub</a></footer>
{PAGE_JS}
</body>
</html>
"""
    if APPLY_BANNER:
        html = APPLY_BANNER(html, current_project="ehs-risk-sem", sections=[], story_href="../story.html")
    return html


def write_search_index(all_modules, all_pages):
    """Record every symbol for cross references and write the search index."""
    SYMBOL_URLS.clear()
    index = []
    for mod_name in all_modules:
        if mod_name not in all_pages:
            continue
        page = all_pages[mod_name]
        index.append({
            "n": mod_name, "m": PKG_NAME, "k": "module",
            "s": first_sentence(page['doc']), "u": f"{mod_name}.html",
        })
        for name in sorted(page['symbols']):
            obj = page['symbols'][name]
            SYMBOL_URLS.setdefault(name, f"{mod_name}.html#{name}")
            index.append({
                "n": name, "m": mod_name, "k": symbol_kind(obj),
                "s": first_sentence(inspect.getdoc(obj)), "u": f"{mod_name}.html#{name}",
            })
    with open(API_DIR / "search-index.js", "w", encoding="utf-8") as f:
        f.write("/* Generated by tools/build_api_docs.py: search index for the API pages. */\n")
        f.write("window.API_INDEX = ")
        f.write(json.dumps(index, ensure_ascii=True, separators=(",", ":")))
        f.write(";\n")


def main():
    """Generate API documentation."""
    API_DIR.mkdir(parents=True, exist_ok=True)

    all_modules = get_all_modules()
    if not all_modules:
        print(f"No modules found in {PKG_PATH}", file=sys.stderr)
        return 1

    all_pages = {}
    for mod_name in all_modules:
        page_data = generate_module_page(mod_name)
        if page_data:
            all_pages[mod_name] = page_data

    write_search_index(all_modules, all_pages)

    for mod_name in all_modules:
        if mod_name not in all_pages:
            continue
        page = all_pages[mod_name]
        count = len(page['symbols'])
        n_cls = sum(1 for o in page['symbols'].values() if inspect.isclass(o))
        content_html = (
            f'<p class="crumbs"><a href="index.html">API</a> / {PKG_NAME}.{mod_name}</p>\n'
            f'<h1>{mod_name}</h1>\n'
            f'<div class="lede">{format_docstring_html(page["doc"])}</div>\n'
            f'<div class="meta"><span class="chip">{count} public symbol{"s" if count != 1 else ""}</span>'
            f'<span class="chip">{n_cls} class{"es" if n_cls != 1 else ""}</span>'
            f'<span class="chip">{PKG_NAME}.{mod_name}</span></div>\n'
            + render_module_content(page)
        )
        html = generate_html_page(
            title=f"API: {mod_name} - ehs-risk-sem",
            description=f"API documentation for {PKG_NAME}.{mod_name}",
            canonical=f"{SITE}/api/{mod_name}.html",
            side=sidebar_html(all_modules, mod_name, page['symbols']),
            content_html=content_html,
        )
        with open(API_DIR / f"{mod_name}.html", 'w', encoding="utf-8") as f:
            f.write(html)
        print(f"Generated {API_DIR / (mod_name + '.html')}")

    # Index page
    pkg = importlib.import_module(PKG_NAME)
    cards = []
    for mod_name in all_modules:
        if mod_name not in all_pages:
            continue
        count = len(all_pages[mod_name]['symbols'])
        summary = inline_markup(first_sentence(all_pages[mod_name]['doc']))
        cards.append(
            f'<li><a href="{mod_name}.html"><b>{mod_name}</b><span>{summary}</span>'
            f'<small>{count} public symbol{"s" if count != 1 else ""}</small></a></li>'
        )
    total = sum(len(all_pages[m]['symbols']) for m in all_pages)
    overview_html = (
        '<p class="crumbs"><a href="../index.html">ehs-risk-sem</a> / API</p>\n'
        '<h1>API Reference</h1>\n'
        f'<div class="lede">{format_docstring_html(inspect.getdoc(pkg) or "")}</div>\n'
        f'<div class="meta"><span class="chip">{len(all_pages)} modules</span>'
        f'<span class="chip">{total} public symbols</span>'
        '<span class="chip">Press / to search</span></div>\n'
        f'<ul class="mods">{"".join(cards)}</ul>'
    )
    html = generate_html_page(
        title="API Reference - ehs-risk-sem",
        description=f"Complete API reference for {PKG_NAME} package",
        canonical=f"{SITE}/api/",
        side=sidebar_html(all_modules),
        content_html=overview_html,
        is_index=True,
    )
    with open(API_DIR / "index.html", 'w', encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {API_DIR / 'index.html'}")

    print(f"\nSummary: {len(all_modules)} modules, {total} public symbols")
    return 0


if __name__ == '__main__':
    sys.exit(main())
