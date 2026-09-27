#!/usr/bin/env python3
"""Check the LLM judge against hand grades.

Usage:
    python hand_grade.py page data/eval/judgments/<file>.json      # writes a grading page, open it in a browser
    python hand_grade.py score data/eval/judgments/<file>.json <grades>.json
    python hand_grade.py review data/eval/judgments/<file>.json <grades>.json  # settle disagreements

page   picks ~20 judged answers stratified by the judge's verdict (every verdict and category shows up,
       plus the known judge mistakes in MUST_INCLUDE) and writes a self-contained HTML page showing the
       question, reference answer, excerpts and answer, but NOT the judge's verdict. Questions are shuffled
       so verdicts don't come in blocks. Grades are kept in the browser's localStorage while you work;
       "Download grades" saves them as JSON.
review writes a page with only the answers where you and the judge disagree, now showing both grades and
       the judge's labels, to pick a settled grade for each. "Download grades" saves the same grades file
       with a "settled" entry added per reviewed answer; the blind grades are kept as they were.
score  compares the downloaded grades with the judge: agreement, Cohen's kappa (agreement beyond what
       chance would give with the same label frequencies), a confusion table and every disagreement.
       With settled grades it also scores the judge against them (blind grade where you both agreed).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from judge_answers import VERDICTS, contradictions, declined

# Judge mistakes and model errors already seen (see CLAUDE.md), so the check covers them.
MUST_INCLUDE = ("q033", "q088", "q060", "q054")
# Per verdict, answerable only; one unanswerable is added on top.
QUOTAS = {"correct": 6, "partial": 5, "incorrect": 5, "refused": 4}


def pick(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    judged = [r for r in rows if "verdict" in r]
    chosen = [r for r in judged if r["id"] in MUST_INCLUDE]
    for verdict, quota in QUOTAS.items():
        pool = [r for r in judged if r["verdict"] == verdict and r["category"] != "unanswerable" and r not in chosen]
        need = quota - sum(r["verdict"] == verdict and r["category"] != "unanswerable" for r in chosen)
        # Round-robin over categories so one big category (paraphrase) doesn't fill the quota.
        by_cat = defaultdict(list)
        for r in rng.sample(pool, len(pool)):
            by_cat[r["category"]].append(r)
        cats = sorted(by_cat, key=lambda c: rng.random())
        while need > 0 and any(by_cat.values()):
            for c in cats:
                if need > 0 and by_cat[c]:
                    chosen.append(by_cat[c].pop())
                    need -= 1
    chosen.append(rng.choice([r for r in judged if r["category"] == "unanswerable" and r not in chosen]))
    rng.shuffle(chosen)
    return chosen


def load(judgments_path: Path) -> tuple[dict, dict, dict]:
    judged = json.loads(judgments_path.read_text(encoding="utf-8"))
    answers = {a["id"]: a for a in json.loads(Path(judged["answers_file"]).read_text(encoding="utf-8"))["answers"]}
    questions = {q["id"]: q for q in map(json.loads, Path("data/eval/questions.jsonl").open(encoding="utf-8"))}
    return judged, answers, questions


def item(row: dict, answers: dict, questions: dict) -> dict:
    return {
        "id": row["id"],
        "category": row["category"],
        "question": questions[row["id"]]["question"],
        "reference": questions[row["id"]]["reference_answer"],
        "answer": answers[row["id"]]["answer"],
        "chunks": [{"id": c["id"], "versions": c.get("versions") or [c["version"]], "url": c["url"],
                    "content": c["content"]} for c in answers[row["id"]]["chunks"]],
    }


def render(data: dict, path: Path) -> Path:
    # "</" inside the embedded JSON would end the <script> element early.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")),
                    encoding="utf-8")
    return path


def write_page(judgments_path: Path, out_dir: Path, seed: int) -> Path:
    judged, answers, questions = load(judgments_path)
    items = [item(r, answers, questions) for r in pick(judged["questions"], seed)]
    data = {"mode": "grade", "judgments_file": str(judgments_path), "seed": seed, "items": items}
    return render(data, out_dir / f"page-{judgments_path.stem}.html")


def write_review(judgments_path: Path, grades_path: Path, out_dir: Path) -> Path:
    judged, answers, questions = load(judgments_path)
    rows = {r["id"]: r for r in judged["questions"]}
    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    items = []
    for i, g in grades["grades"].items():
        if g.get("grade") and g["grade"] != rows[i]["verdict"]:
            grade = rows[i]["grade"]
            items.append(item(rows[i], answers, questions) | {"blind": g, "judge": {
                "verdict": rows[i]["verdict"], "coverage": grade["coverage"], "refused": declined(grade),
                "contradictions": contradictions(grade),
                "unsupported": [c["claim"] for c in grade["claims"] if c["support"] != "supported"]}})
    data = {"mode": "review", "judgments_file": str(judgments_path), "original": grades,
            "items": items}
    return render(data, out_dir / f"review-{grades_path.stem}.html")


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    human, judge = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(human[v] * judge[v] for v in VERDICTS) / n ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def report(title: str, pairs: list[tuple[str, str]]) -> None:
    agree = sum(h == j for h, j in pairs)
    print(f"{title}: {len(pairs)} answers, judge agrees on {agree} ({agree / len(pairs):.0%}), "
          f"Cohen's kappa {kappa(pairs):.2f}")
    # Without partial: does the judge at least get right vs wrong vs refused?
    coarse = [tuple("correct" if v in ("correct", "partial") else v for v in p) for p in pairs]
    print(f"treating partial as correct: agrees on {sum(h == j for h, j in coarse) / len(coarse):.0%}")

    print("\nrows = you, columns = judge")
    print(" " * 11 + "".join(f"{v:>10}" for v in VERDICTS))
    counts = Counter(pairs)
    for h in VERDICTS:
        print(f"{h:>11}" + "".join(f"{counts[h, j]:>10}" for j in VERDICTS))


def score(judgments_path: Path, grades_path: Path) -> None:
    judged = {r["id"]: r for r in json.loads(judgments_path.read_text(encoding="utf-8"))["questions"]}
    grades = json.loads(grades_path.read_text(encoding="utf-8"))
    if grades["judgments_file"] != str(judgments_path):
        print(f"note: grades were made from {grades['judgments_file']}")
    graded = {i: g for i, g in grades["grades"].items() if g.get("grade")}
    if not graded:
        raise SystemExit("no grades in file")
    report("blind", [(g["grade"], judged[i]["verdict"]) for i, g in graded.items()])

    print("\ndisagreements")
    for i, g in sorted(graded.items()):
        if g["grade"] != judged[i]["verdict"]:
            note = f"  -- {g['note']}" if g.get("note") else ""
            print(f"  {i} {judged[i]['category']:16} you={g['grade']:9} judge={judged[i]['verdict']:9}{note}")

    settled = {i: g["settled"] for i, g in graded.items() if g.get("settled", {}).get("grade")}
    if not settled:
        return
    # Where blind grade and judge agreed there was nothing to settle: the blind grade stands.
    final = {i: settled[i]["grade"] if i in settled else g["grade"] for i, g in graded.items()}
    print()
    report("settled", [(final[i], judged[i]["verdict"]) for i in graded])
    ways = Counter("you" if s["grade"] == graded[i]["grade"] else "judge" if s["grade"] == judged[i]["verdict"]
                   else "neither" for i, s in settled.items())
    print(f"\n{len(settled)} settled: your blind grade {ways['you']}, the judge's {ways['judge']}, "
          f"neither {ways['neither']}")
    for i, s in sorted(settled.items()):
        if s["grade"] != judged[i]["verdict"]:
            note = f"  -- {s['note']}" if s.get("note") else ""
            print(f"  judge still wrong: {i} {judged[i]['category']:16} settled={s['grade']:9} "
                  f"judge={judged[i]['verdict']:9}{note}")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand Grading</title>
<style>
:root { --bg:#fafaf9; --card:#fff; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --accent:#2563eb; --code:#f5f5f4; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --card:#292524; --fg:#f5f5f4; --muted:#a8a29e;
  --line:#44403c; --accent:#60a5fa; --code:#1c1917; } }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width:900px; margin:0 auto; padding:16px; }
header { position:sticky; top:0; background:var(--bg); padding:12px 0; border-bottom:1px solid var(--line); z-index:1; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; margin-top:8px; }
.bar div { height:100%; background:var(--accent); width:0; transition:width .2s; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }
.meta { color:var(--muted); font-size:13px; }
h2 { font-size:17px; margin:4px 0 12px; }
h3 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:16px 0 6px; }
.text, pre { white-space:pre-wrap; word-wrap:break-word; }
pre { background:var(--code); border:1px solid var(--line); border-radius:6px; padding:10px; font-size:12.5px;
  max-height:420px; overflow:auto; margin:6px 0 0; }
details { margin:4px 0; } summary { cursor:pointer; font-size:14px; }
.grades { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
.grades button { flex:1 1 120px; padding:10px; border:1px solid var(--line); border-radius:8px;
  background:var(--card); color:var(--fg); font:inherit; cursor:pointer; }
.grades button.on { background:var(--accent); border-color:var(--accent); color:#fff; }
textarea { width:100%; box-sizing:border-box; margin-top:8px; font:inherit; font-size:13px; padding:8px;
  background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
.rubric { font-size:14px; } .rubric dt { font-weight:600; } .rubric dd { margin:0 0 6px 0; color:var(--muted); }
.top { display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
.versus { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:16px; }
.versus > div { border:1px solid var(--line); border-radius:8px; padding:10px; font-size:14px; }
.versus b { font-size:16px; } .versus ul { margin:6px 0 0; padding-left:18px; }
@media (max-width: 600px) { .versus { grid-template-columns:1fr; } }
#dl { padding:8px 14px; border-radius:8px; border:0; background:var(--accent); color:#fff; font:inherit; cursor:pointer; }
</style></head>
<body><main>
<header><div class="top"><strong><span id="mode">Hand grading</span>: <span id="count"></span></strong>
<button id="dl">Download grades</button></div><div class="bar"><div id="fill"></div></div></header>
<div class="card rubric">
<strong id="intro">Grade each answer against the reference answer</strong> <span id="hint">(the judge's verdict is hidden).</span>
<dl>
<dt>correct</dt><dd>States the reference's key points and contradicts none of them.</dd>
<dt>partial</dt><dd>Right as far as it goes, but misses some key points.</dd>
<dt>incorrect</dt><dd>Contradicts the reference or gets none of the key points (e.g. wrong version, invalid SQL that matters).</dd>
<dt>refused</dt><dd>Declines to answer because the excerpts don't cover it, even if it adds a guess after.</dd>
<dt>unanswerable questions</dt><dd>correct if it declines without making things up, otherwise incorrect.</dd>
</dl>
Grades stay in this browser until you download them. Add a note when a grade was a close call.
</div>
<div id="list"></div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const REVIEW = DATA.mode === 'review';
// In review mode `grades` holds the settled grades; the blind ones stay in DATA.original.
const KEY = (REVIEW ? 'hand-review:' : 'hand-grades:') + DATA.judgments_file;
const LABELS = ['correct', 'partial', 'incorrect', 'refused'];
let grades = {};
try { grades = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(grades)); } catch (e) {} };

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  Object.assign(e, attrs || {});
  e.append(...kids);
  return e;
}

function progress() {
  const done = DATA.items.filter(it => grades[it.id] && grades[it.id].grade).length;
  document.getElementById('count').textContent = done + ' / ' + DATA.items.length;
  document.getElementById('fill').style.width = (100 * done / DATA.items.length) + '%';
}

if (REVIEW) {
  document.getElementById('mode').textContent = 'Settling disagreements';
  document.getElementById('intro').textContent = 'Pick the settled grade under this rubric';
  document.getElementById('hint').textContent = '(these are the answers where you and the judge disagreed). '
    + 'If the rubric itself seems wrong, grade by it anyway and say so in the note.';
}

function versus(it) {
  const j = it.judge, list = (title, xs) => xs.length
    ? [el('div', {className: 'meta'}, title), el('ul', {}, ...xs.map(x => el('li', {}, x)))] : [];
  return el('div', {className: 'versus'},
    el('div', {}, el('div', {className: 'meta'}, 'Your blind grade'), el('b', {}, it.blind.grade),
      ...(it.blind.note ? [el('div', {className: 'meta'}, 'Note: ' + it.blind.note)] : [])),
    el('div', {}, el('div', {className: 'meta'}, 'Judge'), el('b', {}, j.verdict),
      el('div', {className: 'meta'}, `coverage ${j.coverage} · refused ${j.refused}`),
      ...list('Contradictions with the reference', j.contradictions),
      ...list('Claims not supported by the excerpts', j.unsupported)));
}

DATA.items.forEach((it, i) => {
  const g = grades[it.id] || (grades[it.id] = {});
  const chunks = it.chunks.map((c, n) => el('details', {},
    el('summary', {}, `[${n + 1}] ${c.id}  (versions ${c.versions.join(', ')})`),
    el('pre', {}, c.content)));
  const buttons = LABELS.map(label => {
    const b = el('button', {className: g.grade === label ? 'on' : ''}, label);
    b.onclick = () => {
      g.grade = label; save(); progress();
      buttons.forEach(x => x.classList.toggle('on', x === b));
    };
    return b;
  });
  const note = el('textarea', {rows: 2, placeholder: 'optional note', value: g.note || ''});
  note.oninput = () => { g.note = note.value; save(); };
  document.getElementById('list').append(el('div', {className: 'card'},
    el('div', {className: 'meta'}, `${i + 1} of ${DATA.items.length} · ${it.id} · ${it.category}`),
    el('h2', {}, it.question),
    el('h3', {}, 'Reference answer'), el('div', {className: 'text'}, it.reference),
    el('h3', {}, 'Excerpts the model saw'), ...chunks,
    el('h3', {}, 'Answer to grade'), el('div', {className: 'text'}, it.answer),
    ...(REVIEW ? [versus(it), el('h3', {}, 'Settled grade')] : []),
    el('div', {className: 'grades'}, ...buttons), note));
});
progress();

document.getElementById('dl').onclick = () => {
  let out = {judgments_file: DATA.judgments_file, seed: DATA.seed, grades};
  if (REVIEW) {
    out = JSON.parse(JSON.stringify(DATA.original));
    for (const [id, s] of Object.entries(grades)) if (s.grade) out.grades[id].settled = s;
  }
  const a = el('a', {download: REVIEW ? 'hand-grades-settled.json' : 'hand-grades.json',
    href: URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}))});
  a.click();
};
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("page", help="write the grading page")
    p.add_argument("judgments", type=Path, help="a file saved by judge_answers.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("data/eval/hand_grades"))
    s = sub.add_parser("score", help="compare downloaded grades with the judge")
    s.add_argument("judgments", type=Path)
    s.add_argument("grades", type=Path, help="the JSON downloaded from the page")
    r = sub.add_parser("review", help="write a page to settle disagreements")
    r.add_argument("judgments", type=Path)
    r.add_argument("grades", type=Path, help="the JSON downloaded from the grading page")
    r.add_argument("--out-dir", type=Path, default=Path("data/eval/hand_grades"))
    args = ap.parse_args()

    if args.cmd == "page":
        print(f"wrote {write_page(args.judgments, args.out_dir, args.seed)}")
    elif args.cmd == "review":
        print(f"wrote {write_review(args.judgments, args.grades, args.out_dir)}")
    else:
        score(args.judgments, args.grades)


if __name__ == "__main__":
    main()
