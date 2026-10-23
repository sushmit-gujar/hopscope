"""Build docs/data.json (the GitHub Pages explorer) from a results directory."""

import json
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/hotpotqa-1000")
summary = json.loads((src / "summary.json").read_text())
perq = json.loads((src / "per_question.json").read_text())
qs = []
for r in perq:
    qs.append({"q": r["question"], "a": r["answer"], "type": r["type"], "gold": r["gold"],
               "m": {m: {"top": v["top"], "f5": v.get("full_support@5"),
                         **({"pred": v["pred"], "em": v.get("em")} if "pred" in v else {})}
                     for m, v in r["methods"].items()}})
out = {**summary, "questions": qs}
Path("docs/data.json").write_text(json.dumps(out, separators=(",", ":")))
print(f"wrote docs/data.json ({Path('docs/data.json').stat().st_size / 1e6:.1f} MB, {len(qs)} q)")
