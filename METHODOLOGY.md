# Methodology, results, and goals

## 1. Question
How well can the umpire's ball/strike call on a taken pitch be predicted, what information
drives it, and how has the called zone changed? The original 2019 project (logistic regression,
~8k pitches from a Kaggle file) reported 82.4% accuracy. This repo rebuilds it on Statcast.

## 2. Data and preprocessing
- Source: Baseball Savant Statcast via `pybaseball.statcast()`, seasons 2019 and 2022–2026
  (2026 through Oct 11, postseason in progress at time of pull).
- Rows kept: `description in {called_strike, ball}` only. Swings, fouls, balls in play, HBP,
  pitchouts are excluded, so every row is a pitch an umpire judged.
- Regular season only (`game_type == "R"`) for all modeling. Spring training (S) is excluded
  because some 2025–26 spring games used the ABS challenge system. Postseason (F/D/L/W) is
  excluded for now; small and a different umpire pool.
- Outlier clip: `plate_x ∈ [-3, 3]` ft, `plate_z ∈ [-1, 6]` ft. Removes a few dozen rows of
  tracking garbage (e.g. plate_z = −57).
- Columns retained: game_date, game_pk, game_type, teams, inning, batter, pitcher,
  fielder_2 (catcher), stand, p_throws, balls, strikes, outs, pitch_type, release_speed,
  release_spin_rate, pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot, zone, description.
- Null rate on every modeling column: 0.0%. Catcher ID is fully populated. Umpire ID is NOT
  in Statcast and requires a separate join (see goals).
- Sizes: 2022–24 train = 1,058,105; 2025–26 test = 708,016; 2019 = 362,950 (RS, post-clip).

## 3. Evaluation protocol
- Out-of-time split: train 2022–2024, test 2025–2026. Never report headline numbers from a
  random split across seasons; umpire rosters, pitch mix, and the zone itself drift.
- Metrics: accuracy at 0.5, log loss, Brier score, ROC AUC. Log loss and Brier matter more
  than accuracy because the demo displays a probability.
- Accuracy is also reported by distance from the batter's own rulebook zone edge, in inches,
  because that is where all remaining error lives.
- Base rate: strike rate is ~33%, so "always ball" = 67%. Any model must be judged against that.

## 4. Models

### Model A — original notebook spec
`logit(strike) ~ px + pz + px² + pz² + balls + strikes`. Six terms, identical to the 2019
notebook's statsmodels formula. Fit with sklearn LogisticRegression (C=10) after standard
scaling; predictions are indistinguishable from an unpenalized MLE fit at this n.

### Model B — batter-normalized zone
Location expressed relative to each batter's zone: `xn = plate_x / 0.83`,
`zn = (plate_z − (sz_top+sz_bot)/2) / ((sz_top−sz_bot)/2)`, so the rulebook zone is the
square [−1,1]². Features: xn, zn, xn², zn², xn·zn, xn⁴, zn⁴, xn²·zn², RHB indicator and its
interactions with xn and xn², balls, strikes, and count × xn² / zn² interactions.

### Results (test set, 708,016 called pitches, 2025–26 regular season)
| model | accuracy | log loss | Brier | AUC |
|---|---|---|---|---|
| A — original 6-term | 92.53% | 0.1685 | 0.0523 | 0.9816 |
| B — normalized + hand | 93.51% | 0.1542 | 0.0465 | 0.9842 |

Accuracy by distance from the batter's zone edge (test set):
| bucket | n | strike rate | acc A | acc B |
|---|---|---|---|---|
| >6 in inside | 33,920 | 1.000 | 1.000 | 1.000 |
| 3–6 in inside | 69,007 | 0.994 | 0.993 | 0.994 |
| 1–3 in inside | 65,823 | 0.934 | 0.926 | 0.934 |
| 0–1 in inside | 37,792 | 0.765 | 0.752 | 0.771 |
| 0–1 in outside | 40,023 | 0.507 | 0.655 | 0.649 |
| 1–3 in outside | 83,716 | 0.183 | 0.770 | 0.811 |
| 3–6 in outside | 120,287 | 0.021 | 0.959 | 0.979 |
| >6 in outside | 257,448 | 0.001 | 0.999 | 0.999 |

Count effect (Model B, RHB, pitch exactly on the outside-corner edge, zone-centre height):
P(strike) = 0.875 at 0-0 · 0.721 at 0-2 · 0.923 at 3-0. Roughly 20 points of zone from count alone.

### Model C — LightGBM (`scripts/gbm.py`)
Fit 2022–23; 2024 split by game into early-stopping and isotonic-calibration halves; test 2025–26.
Params: lr 0.05, 127 leaves, min 500 per leaf, feature fraction 0.9, L2 1, early stopping 100.
Framing score: catcher's mean residual (called_strike − Model B fit on 2022–23), shrunk by
n / (n + 2000); out-of-fold by game (5 folds) for fit rows, from 2022–23 for 2024 rows, from
2022–24 for test rows.

| model | accuracy | log loss | Brier | AUC |
|---|---|---|---|---|
| GBM: zone + count | 94.05% | 0.1403 | 0.0428 | 0.9866 |
| GBM: + pitch | 94.17% | 0.1383 | 0.0424 | 0.9870 |
| GBM: + pitch + framing | 94.13% | 0.1383 | 0.0424 | 0.9870 |

## 5. Why the original 82.4% was so much lower (verified 2026-10-12)
Three checks on Model A:
1. Trained on a random 8,147-row sample of 2022–24 (the notebook's n), tested on 2025–26:
   92.5–92.6% across 5 seeds. Full 1.06M rows: 92.53%. Sample size is not the cause.
2. 2019 Statcast only, random 8,147 train / 354,803 test (the notebook's own protocol):
   91.7–91.8% across 5 seeds.
3. Trained on all of 2019, tested on 2025–26: 92.2%. The called zone is stable across 7 years.

These three checks were one-off runs and are not in `scripts/`; treat them as notes.

Conclusion: the ~10-point gap is attributable to the 2019 source data and label, not the model or
the split. The notebook does show how `Result` was built (cell 11): Kaggle pitch codes
S, C, F, T, L, W, M, Q, R → 1 and B, *B, I, P, H → 0, with balls in play (X, D, E) dropped. So
swinging strikes, fouls and foul tips were labelled 1. The 2019 model predicted "any strike
versus ball", which is a different and noisier target than the umpire's call on a taken pitch,
and 82.4% is not directly comparable with the numbers here. Rerunning the notebook's own label on
the Kaggle file would close this fully; not done.

## 6. Goals and open questions

### Near-term (decided, in order)
1. **Gradient boosting.** Done 2026-10-26 (`scripts/gbm.py`, see README §4): 94.17% / log loss
   0.1383 / AUC 0.9870 with pitch features, gains concentrated within ±3 in of the edge. The
   framing score added nothing measurable. Not done: partial dependence plots, seed spread,
   (the demo moved to the GBM the same day).
2. **Umpire join.** Statcast has no umpire column. Pull home-plate umpire per game_pk from the
   MLB stats API (`/api/v1/game/{game_pk}/boxscore` → officials) and join. Then: per-umpire
   zone maps, umpire consistency ranking, and whether umpire ID beats catcher ID as a feature.
3. **Drift.** Strike rate on called pitches fell monotonically 33.0% (2024) → 32.6% (2025) →
   32.4% (2026). Decompose: is the zone shrinking (same location, fewer strikes) or is pitch
   selection shifting (fewer pitches in the zone taken)? Relevant to the ABS challenge system
   arriving in MLB.
4. **Demo v2.** Export the winning model to ONNX, run in-browser with onnxruntime-web, add
   batter-height slider, handedness toggle, pitch-type dropdown, and (after step 2) an umpire
   dropdown with side-by-side zones. (The demo already runs Model B as of 2026-10-26.)

### Research branch (different question, do after the above)
5. **Sequence model.** Transformer or GRU over the pitch sequence within an at-bat and the
   umpire's prior calls within a game, to test make-up calls, zone fatigue by inning, and
   post-mound-visit tightening. Requires careful sequence reconstruction with no future leakage.
   The GBM from step 1 is the baseline it must beat on the same out-of-time test.

### Explicitly skipped
- MLP and tabular transformers: match GBM at higher cost on this data; only worth it as a
  learning exercise or if per-umpire embeddings are wanted as an artifact.
- Random train/test splits for headline numbers.

### Nice-to-have
- Writeup / LinkedIn post: "the original 82% was a data problem, not a model limit."
- Pull the Kaggle file and close the §5 question.
- Batter-handedness asymmetry chart (lefties get more of the outside corner).
- Postseason-only evaluation once 2026 playoffs finish.

## 7. Changelog
- 2026-10-12: data pull 2022–2026, baselines A/B, sample-size and 2019 replication checks.
- 2026-10-25: repo initialized, methodology written up, demo copied in.
- 2026-10-26: moved into mlb-pitch-classification. `baseline_logit.py` now names its train/test
  years explicitly. Before this it globbed every parquet file, so once 2019 was downloaded the
  "2022–24" training set silently included 2019 (1,421,055 rows; Model B 93.39%, AUC 0.9838).
  The §4 results are the 2022–24-only numbers, reproduced exactly by the fixed script.
  Output saved to `results/baseline_logit.txt`. Re-fetching 2022 from Statcast gave a file
  identical row-for-row to the 2026-10-12 pull.
- 2026-10-26: `baseline_logit.py` writes Model B's raw-scale coefficients to `results/model_b.json`.
  `demo/index.html` now runs Model B instead of the 2019 notebook coefficients, with batter-side
  and zone-size controls. Its JS output matches Python to 5 decimals on spot checks.
- 2026-10-26: shared loading/features moved to `scripts/common.py` (baseline output byte-identical).
  Added `scripts/gbm.py` (Model C, LightGBM) and `results/gbm.{txt,json}`. Demo unchanged.
- 2026-10-26: demo moved from Model B to the "GBM: + pitch" model, exported to ONNX by `gbm.py`
  and run with ONNX Runtime Web (calibration and pitch profiles in `demo/model.json`). Browser
  output matches Python ONNX Runtime to 5 decimals on spot checks. `gbm.py`'s edge table and
  SHAP now describe this model rather than the framing variant.
