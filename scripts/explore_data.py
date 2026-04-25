"""
Step 1: Explore StatsBomb open data — what competitions and matches are available?
Run with:  uv run python explore_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_competitions, get_matches

# ── 1. All available competitions ──────────────────────────────────────────
print("=" * 60)
print("AVAILABLE COMPETITIONS")
print("=" * 60)

comps = get_competitions()
print(comps[["competition_id", "season_id", "competition_name", "season_name"]]
      .sort_values(["competition_name", "season_name"])
      .to_string(index=False))

# ── 2. Identify competitions with rich event data ──────────────────────────
# StatsBomb open data is best covered for:
#   - La Liga (competition_id=11)
#   - Champions League (competition_id=16)
#   - FIFA World Cup (competition_id=43)
#   - NWSL (competition_id=49)
#   - FA Women's Super League (competition_id=37)
FOCUS = [
    (11, "La Liga"),
    (16, "Champions League"),
    (43, "FIFA World Cup"),
]

print("\n" + "=" * 60)
print("MATCH COUNTS FOR KEY COMPETITIONS")
print("=" * 60)
for cid, name in FOCUS:
    seasons = comps[comps["competition_id"] == cid][["season_id", "season_name"]].drop_duplicates()
    for _, s in seasons.iterrows():
        try:
            m = get_matches(cid, int(s["season_id"]))
            print(f"  {name:<25} | season {s['season_name']:<12} | {len(m):>3} matches")
        except Exception as e:
            print(f"  {name:<25} | season {s['season_name']:<12} | ERROR: {e}")

# ── 3. Pick a dataset and show sample matches ──────────────────────────────
# La Liga 2015/16 (season_id=27) is a well-documented season with full 360 data
# available in StatsBomb open data.
TARGET_COMP = 11   # La Liga
TARGET_SEASON = 27  # 2015/16

print("\n" + "=" * 60)
print(f"SAMPLE MATCHES — La Liga season_id={TARGET_SEASON}")
print("=" * 60)

matches = get_matches(TARGET_COMP, TARGET_SEASON)
cols = ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]
print(matches[cols].head(20).to_string(index=False))

print(f"\nTotal matches in this season: {len(matches)}")
print(f"\nColumns available in matches DataFrame:")
print(matches.columns.tolist())

# ── 4. Outcome distribution ─────────────────────────────────────────────────
wins  = (matches["home_score"] > matches["away_score"]).sum()
draws = (matches["home_score"] == matches["away_score"]).sum()
losses = (matches["home_score"] < matches["away_score"]).sum()
print(f"\nOutcome distribution (home perspective):")
print(f"  Home wins : {wins}")
print(f"  Draws     : {draws}")
print(f"  Away wins : {losses}")
