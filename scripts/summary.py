#!/usr/bin/env python3
"""Render an HTML report from state.json.

Usage:
    python summary.py --state <state.json> --out <report.html>

Report: totals, done/review/pending counts, average ms per image, and a
table per image (name, verdict, score, method, ms, attempts, reasons).
Prints ONLY JSON to stdout: {"ok": true, "report": "<absolute>"}.
Logs go to stderr. Exit 0 on success.
"""
import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="HTML report from state.json")
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    state_path = Path(args.state)
    out_path = Path(args.out)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    entries = sorted(state.values(), key=lambda e: e.get("name", ""))
    done = [e for e in entries if e.get("status") == "done"]
    review = [e for e in entries if e.get("status") == "review"]
    pending = [e for e in entries if e.get("status") == "pending"]
    timed = [e for e in done + review if e.get("ms")]
    avg_ms = sum(e["ms"] for e in timed) / len(timed) if timed else 0

    def row(e):
        verdict = e.get("status", "?")
        color = {"done": "#1a7f37", "review": "#b54708",
                 "pending": "#59636e"}.get(verdict, "#000")
        reasons = "<br>".join(html.escape(str(r))
                              for r in (e.get("reasons") or []))
        score = e.get("score")
        return (f"<tr><td>{html.escape(str(e.get('name', '')))}</td>"
                f"<td style='color:{color};font-weight:bold'>{verdict}</td>"
                f"<td>{'' if score is None else score}</td>"
                f"<td>{html.escape(str(e.get('method') or ''))}</td>"
                f"<td>{'' if e.get('ms') is None else round(e['ms'], 1)}</td>"
                f"<td>{e.get('attempts', 0)}</td>"
                f"<td style='font-size:12px'>{reasons}</td></tr>")

    body = "\n".join(row(e) for e in entries)
    page = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Batch debrand report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}
.stats{{display:flex;gap:16px;margin:16px 0}}
.stat{{border:1px solid #ddd;border-radius:8px;padding:12px 18px}}
.stat b{{font-size:24px;display:block}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f6f8fa}}
</style></head><body>
<h1>Báo cáo batch xóa watermark</h1>
<div class="stats">
<div class="stat"><b>{len(entries)}</b>tổng ảnh</div>
<div class="stat"><b style="color:#1a7f37">{len(done)}</b>done</div>
<div class="stat"><b style="color:#b54708">{len(review)}</b>review</div>
<div class="stat"><b>{len(pending)}</b>pending</div>
<div class="stat"><b>{avg_ms:.0f} ms</b>TB / ảnh</div>
</div>
<table><tr><th>Ảnh</th><th>Kết quả</th><th>Score</th><th>Method</th>
<th>ms</th><th>Lần thử</th><th>Lý do</th></tr>
{body}
</table></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    log(f"report written: {out_path} ({len(done)} done, {len(review)} review)")
    print(json.dumps({"ok": True, "report": str(out_path.resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
