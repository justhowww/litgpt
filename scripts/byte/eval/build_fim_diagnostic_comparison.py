#!/usr/bin/env python3
"""Build a tabbed shell around multiple FIM diagnostic sample browsers."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tab",
        action="append",
        required=True,
        metavar="LABEL=REPORT_INDEX",
        help="tab label and an existing diagnostic index.html; repeat for each run",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def parse_tabs(values: list[str], out: Path) -> list[tuple[str, str]]:
    tabs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --tab {value!r}; expected LABEL=REPORT_INDEX")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not label.strip():
            raise SystemExit(f"empty tab label in {value!r}")
        if path.is_dir():
            path = path / "index.html"
        if not path.is_file():
            raise SystemExit(f"diagnostic index is missing: {path}")
        tabs.append((label.strip(), os.path.relpath(path, out.parent)))
    return tabs


def render(tabs: list[tuple[str, str]]) -> str:
    buttons = []
    frames = []
    for index, (label, href) in enumerate(tabs):
        selected = index == 0
        buttons.append(
            f'<button type="button" role="tab" id="tab-{index}" '
            f'aria-controls="panel-{index}" aria-selected="{str(selected).lower()}" '
            f'data-index="{index}" class="{"active" if selected else ""}">'
            f'{html.escape(label)}</button>'
        )
        frames.append(
            f'<section role="tabpanel" id="panel-{index}" aria-labelledby="tab-{index}"'
            f'{"" if selected else " hidden"}>'
            f'<iframe src="{html.escape(href)}" title="{html.escape(label)} diagnostic"></iframe>'
            '</section>'
        )
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FIM repair comparison</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0e1117; --surface:#171c24; --line:#333d4c; --text:#edf2f7; --muted:#a5b0bf; }}
    * {{ box-sizing:border-box; }}
    html, body {{ height:100%; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; display:flex; flex-direction:column; }}
    header {{ display:flex; align-items:center; gap:18px; padding:10px 16px; background:var(--surface); border-bottom:1px solid var(--line); flex-wrap:wrap; }}
    h1 {{ margin:0; font-size:16px; font-weight:600; }}
    [role=tablist] {{ display:flex; gap:6px; flex-wrap:wrap; }}
    [role=tab] {{ color:var(--muted); background:transparent; border:1px solid var(--line); border-radius:5px; padding:7px 11px; font:inherit; }}
    [role=tab].active {{ color:#10151d; background:#e5edf7; border-color:#e5edf7; }}
    main, [role=tabpanel], iframe {{ width:100%; flex:1; min-height:0; }}
    main {{ display:flex; }}
    [role=tabpanel] {{ display:flex; }}
    [role=tabpanel][hidden] {{ display:none; }}
    iframe {{ border:0; background:var(--bg); }}
  </style>
</head>
<body>
  <header>
    <h1>FIM repair comparison</h1>
    <div role="tablist" aria-label="Model and decoding mode">{''.join(buttons)}</div>
  </header>
  <main>{''.join(frames)}</main>
  <script>
    const tabs = [...document.querySelectorAll('[role=tab]')];
    const panels = [...document.querySelectorAll('[role=tabpanel]')];
    function select(index) {{
      tabs.forEach((tab, i) => {{
        const active = i === index;
        tab.classList.toggle('active', active);
        tab.setAttribute('aria-selected', String(active));
        panels[i].hidden = !active;
      }});
      history.replaceState(null, '', `#tab=${{index}}`);
    }}
    tabs.forEach((tab, index) => {{
      tab.addEventListener('click', () => select(index));
      tab.addEventListener('keydown', event => {{
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const delta = event.key === 'ArrowRight' ? 1 : -1;
        const next = (index + delta + tabs.length) % tabs.length;
        select(next);
        tabs[next].focus();
      }});
    }});
    const match = location.hash.match(/tab=(\d+)/);
    if (match && Number(match[1]) < tabs.length) select(Number(match[1]));
  </script>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tabs = parse_tabs(args.tab, args.out.resolve())
    args.out.write_text(render(tabs))
    print(args.out.resolve())


if __name__ == "__main__":
    main()
