"""
Step 1b: Inspect event structure for a single match to plan graph construction.
Run with:  uv run python inspect_events.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_matches, get_events, get_lineups

COMP, SEASON = 11, 27  # La Liga 2015/16

matches = get_matches(COMP, SEASON)
match = matches.iloc[0]
mid = int(match["match_id"])

print(f"Inspecting match {mid}: {match['home_team']} vs {match['away_team']}")
print(f"Score: {match['home_score']} - {match['away_score']}\n")

# ── Events ─────────────────────────────────────────────────────────────────
events = get_events(mid)
print("=" * 60)
print(f"Total events: {len(events)}")
print(f"\nEvent types and counts:")
print(events["type"].value_counts().to_string())

# ── Pass events ─────────────────────────────────────────────────────────────
passes = events[events["type"] == "Pass"].copy()
print(f"\n{'='*60}")
print(f"Pass events: {len(passes)}")
print(f"\nPass columns available:")
print(passes.columns.tolist())

# Check nested pass details
print(f"\nSample pass 'pass' column (first 3):")
for i, row in passes.head(3).iterrows():
    print(f"  {row.get('pass', {})}")

# ── Lineups ─────────────────────────────────────────────────────────────────
lineups = get_lineups(mid)
print(f"\n{'='*60}")
print(f"Teams: {list(lineups.keys())}")

for team, lineup_df in lineups.items():
    print(f"\n{team} lineup columns: {lineup_df.columns.tolist()}")
    print(lineup_df[["player_id", "player_name", "jersey_number"]].head(5).to_string(index=False))

# ── Key pass fields for graph construction ─────────────────────────────────
print(f"\n{'='*60}")
print("KEY FIELDS FOR GRAPH CONSTRUCTION")
print("=" * 60)

# Player, team, outcome fields in pass events
for col in ["player", "team", "pass_recipient", "pass_outcome", "under_pressure",
            "pass_length", "pass_angle", "position", "location", "pass_end_location"]:
    if col in passes.columns:
        sample = passes[col].dropna().iloc[0] if not passes[col].dropna().empty else "N/A"
        print(f"  {col:<25}: {str(sample)[:80]}")

# How many unique players pass in this match per team
home, away = list(lineups.keys())
home_events = events[events["team"] == home]
away_events = events[events["team"] == away]
home_passes = home_events[home_events["type"] == "Pass"]
away_passes = away_events[away_events["type"] == "Pass"]

print(f"\nHome team ({home}):")
print(f"  Passes: {len(home_passes)}, unique passers: {home_passes['player'].nunique()}")
print(f"Away team ({away}):")
print(f"  Passes: {len(away_passes)}, unique passers: {away_passes['player'].nunique()}")

# Check pass recipient to understand edge construction
print(f"\nSample pass events (passer -> recipient, outcome):")
for _, row in home_passes.head(8).iterrows():
    passdict = row.get("pass", {})
    recipient = passdict.get("recipient", {}).get("name", "?") if isinstance(passdict, dict) else "?"
    outcome = passdict.get("outcome", {}).get("name", "Complete") if isinstance(passdict, dict) else "?"
    print(f"  {row['player']:<25} -> {recipient:<25} [{outcome}]")
