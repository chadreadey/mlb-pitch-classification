import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from common import ROOT, load, f_original, f_normalized, edge_dist, EDGE_BINS, EDGE_LABELS

TRAIN_YEARS, TEST_YEARS = [2022, 2023, 2024], [2025, 2026]   # 2019 is deliberately not used here

df = load(TRAIN_YEARS + TEST_YEARS)
y = df.called_strike.values

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
cat = pd.cut(ed, bins=EDGE_BINS, labels=EDGE_LABELS)
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

# --- export Model B on the raw feature scale for demo/index.html ---
sc, lr = m.named_steps["standardscaler"], m.named_steps["logisticregression"]
w = lr.coef_[0] / sc.scale_
names = ["xn", "zn", "xn2", "zn2", "xnzn", "xn4", "zn4", "xn2zn2", "R", "R_xn", "R_xn2",
         "balls", "strikes", "balls_xn2", "strikes_xn2", "balls_zn2", "strikes_zn2"]
out = {"model": "B. batter-normalized zone + hand", "train_years": TRAIN_YEARS, "test_years": TEST_YEARS,
       "n_train": len(tr), "n_test": len(te),
       "test": {"accuracy": round(accuracy_score(yt, p1 > 0.5), 4), "log_loss": round(log_loss(yt, p1), 4),
                "brier": round(brier_score_loss(yt, p1), 4), "auc": round(roc_auc_score(yt, p1), 4)},
       "intercept": float(lr.intercept_[0] - (w * sc.mean_).sum()),
       "coef": dict(zip(names, map(float, w))),
       "mean_sz_top": round(float(tr.sz_top.mean()), 3), "mean_sz_bot": round(float(tr.sz_bot.mean()), 3)}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "model_b.json").write_text(json.dumps(out, indent=2) + "\n")
print("\nwrote results/model_b.json")
