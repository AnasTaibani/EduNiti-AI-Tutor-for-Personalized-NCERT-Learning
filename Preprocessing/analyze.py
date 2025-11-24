#!/usr/bin/env python3
# analyze_passages_stats.py
import json, statistics, re
from pathlib import Path
P = Path("NCERT_passages_hybrid/passage_index.csv")  # adjust if different
if not P.exists():
    print("passage_index.csv not found:", P); raise SystemExit

token_counts = []
by_chapter = {}
with P.open("r", encoding="utf-8") as f:
    import csv
    rd = csv.DictReader(f)
    for r in rd:
        tc = int(r["token_count"])
        token_counts.append(tc)
        ch = (r["grade"], r["subject"], r["chapter"])
        by_chapter.setdefault(ch,0)
        by_chapter[ch]+=1

print("Total passages:", len(token_counts))
print("Token stats — min:", min(token_counts), "25pc:", int(statistics.quantiles(token_counts, n=4)[0]),
      "median:", statistics.median(token_counts), "75pc:", int(statistics.quantiles(token_counts, n=4)[2]),
      "max:", max(token_counts))
print("Avg passages per chapter (sample of chapters):")
import itertools
chap_counts = list(by_chapter.items())
chap_counts_sorted = sorted(chap_counts, key=lambda x: x[1], reverse=True)
for (g,s,c),cnt in chap_counts_sorted[:10]:
    print(f" {g}/{s}/{c}: {cnt} passages")
print("Median chapters with few passages (bottom 10):")
for (g,s,c),cnt in chap_counts_sorted[-10:]:
    print(f" {g}/{s}/{c}: {cnt} passages")
