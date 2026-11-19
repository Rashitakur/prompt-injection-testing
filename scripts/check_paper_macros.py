#!/usr/bin/env python
"""List every \\res{..}{..}{..} used in the paper that the generated macros do not define.

    python scripts/check_paper_macros.py [paper_dir]
A missing value renders as a red ?? in the PDF; usually it means a detector or split was not run.
"""
import re
import sys
from pathlib import Path

paper = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "paper")
used = sorted(set(re.findall(r"\\res\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", (paper / "main.tex").read_text())))
mfile = paper / "generated" / "macros.tex"
defined = set(re.findall(r"\\csname res@([^@]*)@([^@]*)@([^\\]*)\\endcsname", mfile.read_text())) if mfile.exists() else set()
missing = [u for u in used if u not in defined]
print(f"{len(used)} numbers referenced, {len(used) - len(missing)} defined, {len(missing)} missing")
for m in missing:
    print("  MISSING \\res{%s}{%s}{%s}" % m)
sys.exit(1 if missing else 0)
