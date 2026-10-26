import pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
TRAIN_YEARS, TEST_YEARS = [2022, 2023, 2024], [2025, 2026]   # 2019 is deliberately not used here

df = pd.concat([pd.read_parquet(DATA / f"called_pitches_{y}.parquet") for y in TRAIN_YEARS + TEST_YEARS])
df = df[df.game_type == "R"]
df = df[df.plate_x.between(-3, 3) & df.plate_z.between(-1, 6)]
df["year"] = df.game_date.str[:4].astype(int)
y = df.called_strike.values

# --- feature sets ---
def f_original(d):            # the 2019 notebook spec
    return np.c_[d.plate_x, d.plate_z, d.plate_x**2, d.plate_z**2, d.balls, d.strikes]

def f_normalized(d):          # location relative to the batter's own zone + handedness
    zc = (d.sz_top + d.sz_bot) / 2; zh = (d.sz_top - d.sz_bot) / 2
    zn = (d.plate_z - zc) / zh                 # -1..1 inside the vertical zone
    xn = d.plate_x / 0.83                      # -1..1 inside the horizontal zone
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

tr, te = df[df.year.isin(TRAIN_YEARS)], df[df.year.isin(TEST_YEARS)]
print(f"train {len(tr):,} (2022-24 regular season) | test {len(te):,} (2025-26 regular season)\n")

def run(name, fn):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))
    m.fit(fn(tr), tr.called_strike)
    p = m.predict_proba(fn(te))[:, 1]
    print(f"{name:<34} acc {accuracy_score(te.called_strike, p>0.5):.4f}  logloss {log_loss(te.called_strike, p):.4f}  "
          f"brier {brier_score_loss(te.called_strike, p):.4f}  auc {roc_auc_score(te.called_strike, p):.4f}")
    return p

p0 = run("A. original spec (raw px/pz)", f_original)
p1 = run("B. batter-normalized zone + hand", f_normalized)

# --- where the gain lives: accuracy by distance from the batter's zone edge ---
ed = edge_dist(te)
bins = [-99, -6, -3, -1, 0, 1, 3, 6, 99]
labels = ["> 6in inside", "3-6in inside", "1-3in inside", "0-1in inside", "0-1in outside", "1-3in outside", "3-6in outside", "> 6in outside"]
cat = pd.cut(ed, bins=bins, labels=labels)
yt = te.called_strike.values
tab = pd.DataFrame({"bucket": cat, "n": 1, "strike_rate": yt, "acc_A": (p0 > 0.5) == yt, "acc_B": (p1 > 0.5) == yt}) \
        .groupby("bucket", observed=True).agg(n=("n","sum"), strike_rate=("strike_rate","mean"), acc_A=("acc_A","mean"), acc_B=("acc_B","mean"))
tab["n"] = tab.n.map("{:,}".format)
print("\naccuracy by distance from the batter's own zone edge (test set):")
print(tab.round(3).to_string())

# --- what the normalized model says about count ---
print("\nP(strike) for a pitch exactly on the outside-corner edge (xn=1, zn=0), RHB, by count:")
m = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000)).fit(f_normalized(tr), tr.called_strike)
rows = []
for b in range(4):
    for s in range(3):
        d = pd.DataFrame({"plate_x":[0.83], "plate_z":[2.5], "sz_top":[3.5], "sz_bot":[1.5], "stand":["R"], "balls":[b], "strikes":[s]})
        rows.append((f"{b}-{s}", m.predict_proba(f_normalized(d))[0,1]))
print(pd.DataFrame(rows, columns=["count","p_strike"]).set_index("count").T.round(3).to_string())
