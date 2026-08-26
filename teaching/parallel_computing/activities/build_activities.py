#!/usr/bin/env python3
"""Turn an activity *with* its solution into the *student* skeleton.

Activities are deliberately tiny (one hole, one payoff, ~5 min) and **not graded**. Each
activity is written once, WITH the answer, wrapped in a solution marker; this script strips
the answer to a friendly stub so the student fills one blank and immediately sees the number.

  python3 build_activities.py activities/1_warmup_python.ipynb
  python3 build_activities.py activities/1_warmup_julia.jl
      -> activities/release/<name>_todo.<ext>

Marker (carries the hint shown to the student):

  Python / Jupyter code cells:
      ## SOLUTION: sum x with a plain for loop ##
      s = 0.0
      for xi in x: s += xi
      return s
      ## END ##

  Julia scripts:
      #= SOLUTION: sum x with a plain loop =#
      ...
      #= END =#

Everything OUTSIDE the markers is kept verbatim — setup, the timing/print that reveals the
payoff, the comments. Only the answer between the markers becomes:

      # ✏️ your turn: <hint>
      <a stub that fails loudly until you replace it>

The instructor file (with the answer) is committed; the generated skeleton lives in
activities/release/ and is gitignored (regenerate anytime). No dependencies, no nbgrader.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.join(HERE, "release")

PY_BEGIN = re.compile(r"^(\s*)##\s*SOLUTION\s*:?\s*(.*?)\s*##\s*$")
PY_END = re.compile(r"^\s*##\s*END\s*##\s*$")
JL_BEGIN = re.compile(r"^(\s*)#=\s*SOLUTION\s*:?\s*(.*?)\s*=#\s*$")
JL_END = re.compile(r"^\s*#=\s*END\s*=#\s*$")


def strip(text, begin_re, end_re, fail_stmt):
    """Replace each SOLUTION..END block by a hinted stub. Returns (new_text, n_holes)."""
    lines, out, i, n = text.split("\n"), [], 0, 0
    while i < len(lines):
        m = begin_re.match(lines[i])
        if m:
            n += 1
            indent, hint = m.group(1), (m.group(2) or "complete this")
            out.append(f"{indent}# ✏️ your turn: {hint}")
            out.append(f"{indent}{fail_stmt}")
            i += 1
            while i < len(lines) and not end_re.match(lines[i]):
                i += 1
            i += 1  # skip END
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), n


def as_text(source):
    return source if isinstance(source, str) else "".join(source)


def as_lines(text):
    p = text.split("\n")
    return [x + "\n" for x in p[:-1]] + [p[-1]]


def build_ipynb(path):
    nb = json.load(open(path))
    holes = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        new, k = strip(as_text(c["source"]), PY_BEGIN, PY_END,
                       'raise NotImplementedError("remove this line and write your code")')
        if k:
            c["source"] = as_lines(new)
            c["outputs"] = []
            c["execution_count"] = None
            holes += k
        elif isinstance(c["source"], list):
            c["outputs"] = []
            c["execution_count"] = None
    return nb, holes


def build_jl(path):
    return strip(open(path).read(), JL_BEGIN, JL_END,
                 'error("remove this line and write your code")')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="activity file with its solution (.ipynb or .jl)")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    os.makedirs(RELEASE, exist_ok=True)
    base = os.path.basename(args.source)

    if args.source.endswith(".jl"):
        text, holes = build_jl(args.source)
        out = args.out or os.path.join(RELEASE, base.replace(".jl", "_todo.jl"))
        open(out, "w").write(text)
    else:
        nb, holes = build_ipynb(args.source)
        out = args.out or os.path.join(RELEASE, base.replace(".ipynb", "_todo.ipynb"))
        json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)

    if holes == 0:
        raise SystemExit(f"⚠  no SOLUTION..END block found in {args.source} — nothing to strip")
    print(f"{holes} hole(s) -> {out}")


if __name__ == "__main__":
    main()
