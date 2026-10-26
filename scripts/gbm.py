"""Gradient boosting (LightGBM) on the called-pitch data, with pitch features and a catcher framing score.

Splits: fit on 2022-23; 2024 is split by game into an early-stopping half and an isotonic-calibration
half; test on 2025-26, the same test seasons as baseline_logit.py.
"""
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from common import ROOT, load, zone_coords, f_normalized, edge_dist, EDGE_BINS, EDGE_LABELS

FIT_YEARS, CAL_YEARS, TEST_YEARS = [2022, 2023], [2024], [2025, 2026]
SHRINK = 2000      # framing: a catcher's residual is shrunk toward 0 by n / (n + SHRINK) called pitches
FOLDS = 5          # framing for fit rows is out-of-fold by game
PITCH_TYPES = ["FF", "SI", "SL", "CH", "FC", "CU", "ST", "FS", "KC", "SV"]   # everything else -> "OTH"

df = load(FIT_YEARS + CAL_YEARS + TEST_YEARS).reset_index(drop=True)
fit, cal, te = (df[df.year.isin(ys)] for ys in (FIT_YEARS, CAL_YEARS, TEST_YEARS))
es, iso = cal[cal.game_pk % 2 == 0], cal[cal.game_pk % 2 == 1]
print(f"fit {len(fit):,} (2022-23) | early-stop {len(es):,} + calibrate {len(iso):,} (2024) | test {len(te):,} (2025-26)\n")

# --- catcher framing: called-strike surplus over a location + count expectation (Model B) ---
logit = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))
logit.fit(f_normalized(fit), fit.called_strike)
df["resid"] = df.called_strike - logit.predict_proba(f_normalized(df))[:, 1]

def framing_from(rows):
    g = rows.groupby("fielder_2").resid.agg(["sum", "count"])
    return g["sum"] / (g["count"] + SHRINK)

df["framing"] = 0.0
fold = fit.game_pk % FOLDS
for k in range(FOLDS):
    score = framing_from(df.loc[fit.index[fold != k]])
    idx = fit.index[fold == k]
    df.loc[idx, "framing"] = df.loc[idx, "fielder_2"].map(score).fillna(0).values
df.loc[cal.index, "framing"] = df.loc[cal.index, "fielder_2"].map(framing_from(df.loc[fit.index])).fillna(0).values
df.loc[te.index, "framing"] = df.loc[te.index, "fielder_2"].map(framing_from(df.loc[fit.index.union(cal.index)])).fillna(0).values

# --- features ---
xn, zn = zone_coords(df)
X = pd.DataFrame({
    "xn": xn, "zn": zn, "sz_top": df.sz_top, "sz_bot": df.sz_bot,
    "batter_R": (df.stand == "R").astype(int), "balls": df.balls, "strikes": df.strikes,
    "pitcher_R": (df.p_throws == "R").astype(int),
    "pitch_type": pd.Categorical(df.pitch_type.where(df.pitch_type.isin(PITCH_TYPES), "OTH"), categories=PITCH_TYPES + ["OTH"]),
    "release_speed": df.release_speed, "spin_rate": df.release_spin_rate, "pfx_x": df.pfx_x, "pfx_z": df.pfx_z,
    "framing": df.framing})
ZONE = ["xn", "zn", "sz_top", "sz_bot", "batter_R", "balls", "strikes"]
PITCH = ["pitcher_R", "pitch_type", "release_speed", "spin_rate", "pfx_x", "pfx_z"]
VARIANTS = {"GBM: zone + count": ZONE, "GBM: + pitch": ZONE + PITCH, "GBM: + pitch + framing": ZONE + PITCH + ["framing"]}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
              feature_fraction=0.9, lambda_l2=1.0, seed=0, verbose=-1)

yt = te.called_strike.values
def metrics(p):
    return dict(accuracy=accuracy_score(yt, p > 0.5), log_loss=log_loss(yt, p), brier=brier_score_loss(yt, p), auc=roc_auc_score(yt, p))
def line(name, m):
    print(f"{name:<34} acc {m['accuracy']:.4f}  logloss {m['log_loss']:.4f}  brier {m['brier']:.4f}  auc {m['auc']:.4f}", flush=True)

# reference: Model B as reported (trained on all of 2022-24)
ref_tr = df[df.year.isin(FIT_YEARS + CAL_YEARS)]
ref = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000)).fit(f_normalized(ref_tr), ref_tr.called_strike)
preds = {"Model B (logistic, 2022-24)": ref.predict_proba(f_normalized(te))[:, 1]}
results = {"Model B (logistic, 2022-24)": metrics(preds["Model B (logistic, 2022-24)"])}
line("Model B (logistic, 2022-24)", results["Model B (logistic, 2022-24)"])

boosters = {}
for name, cols in VARIANTS.items():
    b = lgb.train(PARAMS, lgb.Dataset(X.loc[fit.index, cols], fit.called_strike),
                  num_boost_round=5000, valid_sets=[lgb.Dataset(X.loc[es.index, cols], es.called_strike)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    ir = IsotonicRegression(out_of_bounds="clip").fit(b.predict(X.loc[iso.index, cols]), iso.called_strike)
    p = ir.predict(b.predict(X.loc[te.index, cols]))
    preds[name], boosters[name] = p, (b, cols, ir)
    results[name] = metrics(p) | {"rounds": b.best_iteration}
    line(name + f" ({b.best_iteration} rounds)", results[name])

# --- where the gain lives ---
cat = pd.cut(edge_dist(te), bins=EDGE_BINS, labels=EDGE_LABELS)
best = "GBM: + pitch"            # the headline and demo model; framing is an ablation
tab = pd.DataFrame({"bucket": cat, "n": 1, "strike_rate": yt,
                    "acc_B": (preds["Model B (logistic, 2022-24)"] > 0.5) == yt, "acc_GBM": (preds[best] > 0.5) == yt}) \
        .groupby("bucket", observed=True).agg(n=("n", "sum"), strike_rate=("strike_rate", "mean"), acc_B=("acc_B", "mean"), acc_GBM=("acc_GBM", "mean"))
edge = tab.copy()
tab["n"] = tab.n.map("{:,}".format)
print(f"\naccuracy by distance from the batter's zone edge (test set), Model B vs {best}:")
print(tab.round(3).to_string())

# --- what the full model uses: mean |SHAP| on a 100k test sample ---
b, cols, _ = boosters[best]
sample = X.loc[te.index, cols].sample(100_000, random_state=0)
contrib = b.predict(sample, pred_contrib=True)[:, :-1]
shap = pd.Series(np.abs(contrib).mean(0), index=cols).sort_values(ascending=False)
print(f"\nmean |SHAP| (log-odds), {best}:")
print(shap.round(3).to_string())

# --- pitch type at the edge (data, not model): called-strike rate within 1 inch either side ---
ed = edge_dist(te)
band = te[(ed > -1) & (ed <= 1)]
pt = band.assign(pt=band.pitch_type.where(band.pitch_type.isin(PITCH_TYPES), "OTH")).groupby("pt").called_strike.agg(["mean", "count"])
pt = pt[pt["count"] >= 2000].sort_values("mean")
print("\ncalled-strike rate within 1 inch of the zone edge, by pitch type (test set, n >= 2,000):")
print(pt.round(3).to_string())

# --- framing spread: catchers with >= 5,000 called pitches in 2022-24 ---
pool = df.loc[fit.index.union(cal.index)]
fr = pool.groupby("fielder_2").resid.agg(["sum", "count"])
fr = fr[fr["count"] >= 5000]
fr["per_1000"] = 1000 * fr["sum"] / (fr["count"] + SHRINK)
q = fr.per_1000.quantile([0, .1, .5, .9, 1]).round(1)
print(f"\nframing, called strikes above expectation per 1,000 called pitches (2022-24, {len(fr)} catchers with >= 5,000):")
print(q.to_string())

out = {"splits": {"fit": FIT_YEARS, "early_stop_and_calibrate": CAL_YEARS, "test": TEST_YEARS,
                  "n_fit": len(fit), "n_early_stop": len(es), "n_calibrate": len(iso), "n_test": len(te)},
       "params": PARAMS, "framing_shrink": SHRINK,
       "test": {k: {m: round(float(v), 4) for m, v in r.items()} for k, r in results.items()},
       "edge_accuracy": {str(k): {"n": int(r.n), "strike_rate": round(r.strike_rate, 3), "acc_B": round(r.acc_B, 3), "acc_GBM": round(r.acc_GBM, 3)} for k, r in edge.iterrows()},
       "mean_abs_shap": shap.round(4).to_dict(),
       "edge_strike_rate_by_pitch_type": {k: round(v, 3) for k, v in pt["mean"].items()},
       "framing_per_1000_quantiles": {str(k): v for k, v in q.items()}}

# --- export the demo model: "GBM: + pitch" (framing needs a catcher, which the demo doesn't ask for) ---
import onnxmltools, onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType
DEMO = "GBM: + pitch"
b, cols, ir = boosters[DEMO]
def as_float(frame):          # ONNX input: float32 matrix, pitch type as its category code
    f = frame[cols].copy(); f["pitch_type"] = f.pitch_type.cat.codes
    return f.to_numpy(np.float32)
onx = onnxmltools.convert_lightgbm(b, initial_types=[("input", FloatTensorType([None, len(cols)]))], target_opset=15, zipmap=False)
for o in onx.graph.output:   # the converter declares the label output as length 1; make the batch dim symbolic
    o.type.tensor_type.shape.dim[0].dim_param = "N"
(ROOT / "demo" / "model.onnx").write_bytes(onx.SerializeToString())
chk = X.loc[te.index, cols].sample(20_000, random_state=1)
sess = ort.InferenceSession(onx.SerializeToString())
parity = float(np.abs(sess.run(None, {"input": as_float(chk)})[1][:, 1] - b.predict(chk)).max())
print(f"\nONNX export of {DEMO}: {b.num_trees()} trees, max |ONNX - LightGBM| on 20k test rows = {parity:.1e}")

# typical speed / spin / movement per pitcher hand and pitch type (2022-23 medians)
prof = df.loc[fit.index].assign(pt=X.loc[fit.index, "pitch_type"].astype(str), hand=df.loc[fit.index, "p_throws"]) \
         .groupby(["hand", "pt"])[["release_speed", "release_spin_rate", "pfx_x", "pfx_z"]].median().round(2)
profiles = {h: {pt: dict(zip(["release_speed", "spin_rate", "pfx_x", "pfx_z"], map(float, r))) for (hh, pt), r in prof.iterrows() if hh == h}
            for h in ("R", "L")}

# count grid with the demo model: RHP four-seamer, RHB, outside edge (xn = 1), mid-zone, mean zone
top, bot = float(fit.sz_top.mean()), float(fit.sz_bot.mean())
ff = profiles["R"]["FF"]
grid = pd.DataFrame([dict(xn=1.0, zn=0.0, sz_top=top, sz_bot=bot, batter_R=1, balls=bb, strikes=ss, pitcher_R=1,
                          pitch_type=pd.Categorical(["FF"], categories=PITCH_TYPES + ["OTH"])[0],
                          release_speed=ff["release_speed"], spin_rate=ff["spin_rate"], pfx_x=ff["pfx_x"], pfx_z=ff["pfx_z"])
                     for bb in range(4) for ss in range(3)])
grid["pitch_type"] = pd.Categorical(grid.pitch_type, categories=PITCH_TYPES + ["OTH"])
pg = ir.predict(b.predict(grid[cols]))
count_grid = {f"{bb}-{ss}": round(float(v), 3) for (bb, ss), v in zip([(bb, ss) for bb in range(4) for ss in range(3)], pg)}
print("\nP(called strike), demo model, RHP four-seamer to a RHB on the outside edge at mid-zone, by count:")
print(pd.Series(count_grid).to_frame("p").T.to_string())

meta = {"model": DEMO, "features": cols, "pitch_types": PITCH_TYPES + ["OTH"], "trees": b.num_trees(),
        "test": {m: round(float(v), 4) for m, v in results[DEMO].items()}, "onnx_parity_max_abs": parity,
        "isotonic": {"x": [round(float(v), 6) for v in ir.X_thresholds_], "y": [round(float(v), 6) for v in ir.y_thresholds_]},
        "zone_sizes": {"small": [3.12, 1.44], "avg": [round(top, 3), round(bot, 3)], "tall": [3.64, 1.75]},
        "profiles": profiles, "count_grid_rhp_ff_rhb_outside_edge": count_grid}
(ROOT / "demo" / "model.json").write_text(json.dumps(meta, indent=1) + "\n")
out["demo_count_grid"] = count_grid
(ROOT / "results" / "gbm.json").write_text(json.dumps(out, indent=2) + "\n")
print("\nwrote results/gbm.json, demo/model.onnx, demo/model.json")
