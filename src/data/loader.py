"""
StatsBomb data loader.

Loads competitions, matches, lineups, and events from the StatsBomb open data API.
All data is cached locally under data/raw/ to avoid repeat network calls.
"""

import json
import os
import pickle
from pathlib import Path

import pandas as pd
from statsbombpy import sb

# Project root is two levels up from this file (src/data/loader.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Competitions & matches
# ---------------------------------------------------------------------------

def get_competitions() -> pd.DataFrame:
    """Return all StatsBomb open-data competitions as a DataFrame."""
    cache = RAW_DIR / "competitions.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    comps = sb.competitions()
    comps.to_pickle(cache)
    return comps


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a competition/season, cached to disk."""
    cache = RAW_DIR / f"matches_{competition_id}_{season_id}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches.to_pickle(cache)
    return matches


# ---------------------------------------------------------------------------
# Match-level data
# ---------------------------------------------------------------------------

def get_lineups(match_id: int) -> dict:
    """Return lineups dict {team_name: DataFrame} for a match, cached."""
    cache = RAW_DIR / f"lineups_{match_id}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    lineups = sb.lineups(match_id=match_id)
    with open(cache, "wb") as f:
        pickle.dump(lineups, f)
    return lineups


def get_events(match_id: int) -> pd.DataFrame:
    """Return all events for a match as a flat DataFrame, cached."""
    cache = RAW_DIR / f"events_{match_id}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    events = sb.events(match_id=match_id)
    events.to_pickle(cache)
    return events


# ---------------------------------------------------------------------------
# Bulk loading helpers
# ---------------------------------------------------------------------------

def load_competition_matches(
    competition_id: int,
    season_id: int,
    max_matches: int | None = None,
) -> list[dict]:
    """
    Load events + lineups for every match in a competition/season.

    Returns a list of dicts, one per match:
        {
            "match_id": int,
            "home_team": str,
            "away_team": str,
            "home_score": int,
            "away_score": int,
            "events": pd.DataFrame,
            "lineups": dict,
        }
    """
    matches = get_matches(competition_id, season_id)
    if max_matches is not None:
        matches = matches.head(max_matches)

    records = []
    for _, row in matches.iterrows():
        mid = int(row["match_id"])
        record = {
            "match_id": mid,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "events": get_events(mid),
            "lineups": get_lineups(mid),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Outcome label
# ---------------------------------------------------------------------------

def match_outcome_label(home_score: int, away_score: int) -> int:
    """
    Return outcome from home team perspective.
        0 = home win
        1 = draw
        2 = away win
    """
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    else:
        return 2
