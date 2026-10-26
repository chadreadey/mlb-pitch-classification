"""Download Statcast called pitches (called_strike / ball) per season to parquet.

Usage: python scripts/fetch_statcast.py 2022 2023 2024 2025 2026
Seasons are fetched in ~2-week chunks (pybaseball does this internally) and cached.
"""
import sys, datetime as dt, pathlib
import pandas as pd
from pybaseball import statcast, cache

cache.enable()
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

KEEP = ["game_date","game_pk","game_type","home_team","away_team","inning","inning_topbot",
        "batter","pitcher","fielder_2","stand","p_throws","balls","strikes","outs_when_up",
        "pitch_type","release_speed","release_spin_rate","pfx_x","pfx_z","plate_x","plate_z",
        "sz_top","sz_bot","zone","description","at_bat_number","pitch_number"]

TODAY = dt.date.today()
for year in map(int, sys.argv[1:]):
    start = f"{year}-03-15"
    end = f"{year}-11-10" if year < TODAY.year else TODAY.isoformat()
    print(f"[{year}] fetching {start} -> {end}", flush=True)
    df = statcast(start_dt=start, end_dt=end, verbose=False)
    print(f"[{year}] raw pitches: {len(df):,}", flush=True)
    df = df[df["description"].isin(["called_strike", "ball"])]
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["called_strike"] = (df["description"] == "called_strike").astype("int8")
    df = df.dropna(subset=["plate_x","plate_z","sz_top","sz_bot"])
    path = OUT / f"called_pitches_{year}.parquet"
    df.to_parquet(path, index=False)
    print(f"[{year}] called pitches kept: {len(df):,} -> {path.name} "
          f"(strike rate {df.called_strike.mean():.3f}, last date {df.game_date.max()})", flush=True)
print("done", flush=True)
