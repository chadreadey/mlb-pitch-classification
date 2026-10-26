"""Data loading and features shared by baseline_logit.py and gbm.py."""
import pathlib, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load(years):
    df = pd.concat([pd.read_parquet(DATA / f"called_pitches_{y}.parquet") for y in years])
    df = df[df.game_type == "R"]
    df = df[df.plate_x.between(-3, 3) & df.plate_z.between(-1, 6)]
    df["year"] = df.game_date.str[:4].astype(int)
    return df

def zone_coords(d):           # location relative to the batter's own zone: rulebook zone = [-1, 1]^2
    zc = (d.sz_top + d.sz_bot) / 2; zh = (d.sz_top - d.sz_bot) / 2
    return d.plate_x / 0.83, (d.plate_z - zc) / zh

def f_original(d):            # the 2019 notebook spec
    return np.c_[d.plate_x, d.plate_z, d.plate_x**2, d.plate_z**2, d.balls, d.strikes]

def f_normalized(d):          # Model B: batter-relative zone + handedness + count
    xn, zn = zone_coords(d)
    R = (d.stand == "R").astype(float)
    return np.c_[xn, zn, xn**2, zn**2, xn*zn, xn**4, zn**4, (xn**2)*(zn**2),
                 R, R*xn, R*xn**2, d.balls, d.strikes,
                 d.balls*xn**2, d.strikes*xn**2, d.balls*zn**2, d.strikes*zn**2]

def edge_dist(d):             # inches outside (+) / inside (-) the batter's rulebook zone
    px, pz, top, bot = d.plate_x.values, d.plate_z.values, d.sz_top.values, d.sz_bot.values
    dx = np.clip(np.abs(px) - 0.83, 0, None)
    dz = np.clip(np.maximum(bot - pz, pz - top), 0, None)
    out = np.sqrt(dx**2 + dz**2)
    inside = np.clip(np.minimum.reduce([0.83 - np.abs(px), pz - bot, top - pz]), 0, None)
    return np.where(out > 0, out, -inside) * 12

EDGE_BINS = [-99, -6, -3, -1, 0, 1, 3, 6, 99]
EDGE_LABELS = ["> 6in inside", "3-6in inside", "1-3in inside", "0-1in inside",
               "0-1in outside", "1-3in outside", "3-6in outside", "> 6in outside"]
