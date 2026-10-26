# MLB ball/strike classification: from a 2019 sample to 1.8M Statcast calls

This project began in 2023 as a group project with **Carli Kelley, Chad Readey and Garner Hasler**. That work fit a logistic regression on about 8,000 pitches from the 2019 season and is kept unchanged in [`MLB_Python_Code_Report.ipynb`](MLB_Python_Code_Report.ipynb). Carli Kelley's original repository is [here](https://github.com/carli-kelley/mlb-pitch-classification); I host this copy with permission.

The 2026 rebuild (`scripts/`, `METHODOLOGY.md`) is my own work. Write-up: [How much does the count move the strike zone?](https://chadreadey.github.io/mlb/writeup.html). It asks a narrower question, **will the umpire call this taken pitch a strike?**, and answers it on every called pitch in Statcast from 2022 to 2026 with an out-of-time test.

## 1. What the rebuild changes

- **Target.** Only pitches the umpire judged: `called_strike` or `ball`. The 2019 notebook labelled swinging strikes, fouls and foul tips as 1 too (cell 11), so it was predicting "any strike versus ball". Its 82.4% accuracy is not directly comparable with the numbers below.
- **Data.** Baseball Savant Statcast via `pybaseball`, regular season only. That gives 1,058,105 called pitches in 2022–24 for training and 708,016 in 2025–26 for testing. Spring training is excluded because some 2025–26 spring games used the ABS challenge system. A handful of tracking errors are clipped (`plate_x` outside ±3 ft, `plate_z` outside −1 to 6 ft).
- **Split.** Train on 2022–24 and test on 2025–26. Umpire rosters, pitch mix and the zone itself drift, so a random split across seasons would flatter the model.

## 2. Method

Both models are logistic regressions with standard scaling (scikit-learn, C=10).

- **Model A: the 2019 feature set.** `px, pz, px², pz², balls, strikes`, the same six terms as the notebook's statsmodels formula, refit on the new data.
- **Model B: location relative to the batter's own zone.** Statcast records each batter's zone top and bottom (`sz_top`, `sz_bot`). Location is rescaled so the rulebook zone is the square [−1, 1]²: `xn = plate_x / 0.83` ft (half the plate plus a ball), `zn = (plate_z − zone centre) / zone half-height`. Features are xn, zn, their squares and fourth powers, the cross terms xn·zn and xn²·zn², batter handedness with its interactions with xn and xn², balls, strikes, and count × xn² / zn² interactions. That makes 17 terms.

## 3. Results

Test set: 708,016 called pitches, 2025–26 regular season. About 67% are balls, so "always ball" scores about 67%.

| Model | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| A: 2019 feature set | 92.53% | 0.1685 | 0.0523 | 0.9816 |
| B: batter-relative zone + count | **93.51%** | **0.1542** | **0.0465** | **0.9842** |

How to read this:
- **The 2019 features were already most of the way there.** Refit on clean called-pitch data, the six-term model scores 92.5%. Most of the jump from 82.4% comes from the cleaner target and better data, not from a cleverer model.
- **Batter-relative location adds about one point.** It also gives a better-calibrated probability (lower log loss and Brier), which matters more than accuracy because the demo shows a probability.
- **All the remaining error is at the edge of the zone.** Accuracy by distance from the batter's rulebook zone edge:

| Distance from zone edge | Pitches | Strike rate | Acc. A | Acc. B |
|---|---|---|---|---|
| > 6 in inside | 33,920 | 1.000 | 100.0% | 100.0% |
| 3–6 in inside | 69,007 | 0.994 | 99.3% | 99.4% |
| 1–3 in inside | 65,823 | 0.934 | 92.6% | 93.4% |
| 0–1 in inside | 37,792 | 0.765 | 75.2% | 77.1% |
| 0–1 in outside | 40,023 | 0.507 | 65.5% | 64.9% |
| 1–3 in outside | 83,716 | 0.183 | 77.0% | 81.1% |
| 3–6 in outside | 120,287 | 0.021 | 95.9% | 97.9% |
| > 6 in outside | 257,448 | 0.001 | 99.9% | 99.9% |

Within an inch outside the rulebook zone, umpires call about half the pitches strikes, and neither model does better than about 65%. Model B's gain comes from 1–6 inches outside, not from the closest pitches.

- **The count moves the zone.** For a right-handed batter and a pitch exactly on the outside edge at mid-zone height, Model B gives P(strike) = 0.875 at 0-0, 0.721 at 0-2 and 0.923 at 3-0. That is about 20 points from the count alone.

The full script output is in [`results/baseline_logit.txt`](results/baseline_logit.txt). [`METHODOLOGY.md`](METHODOLOGY.md) has the data rules, the evaluation protocol, extra checks on the 2019 gap, and next steps.

## Limitations

- **Linear model, hand-built features.** Both models are logistic regressions. Pitch type, movement, catcher framing and umpire identity are not used yet; gradient boosting on those is the planned next step.
- **No umpire ID.** Statcast doesn't include the home-plate umpire, so per-umpire zones are out of scope until that join is built.
- **Accuracy at 0.5.** About a third of test pitches are strikes. Log loss and Brier are the better measures for a probability; accuracy is reported because the 2019 notebook used it.
- **Unscripted checks.** The sample-size and 2019-only checks in METHODOLOGY.md §5 were one-off runs and are not in `scripts/`.
- **Demo zone sizes.** [`demo/index.html`](demo/index.html), the interactive strike zone, runs Model B with coefficients copied from `results/model_b.json`. It offers three batter zones (10th percentile, mean and 90th percentile of 2022–24 `sz_top`/`sz_bot`), not every batter's.
- **2026 data.** The 2026 season was pulled on 2026-10-11, after the regular season ended (postseason is excluded anyway).

## Data and terms

- **Source.** [Baseball Savant](https://baseballsavant.mlb.com/) Statcast data, downloaded through [`pybaseball`](https://github.com/jldbc/pybaseball). Statcast data belongs to MLB Advanced Media and is subject to [MLB's terms of use](https://www.mlb.com/official-information/terms-of-use).
- **Raw data is not committed.** `scripts/fetch_statcast.py` downloads it and keeps only the called pitches and the columns used (`data/`, gitignored). Re-fetching 2022 on 2026-10-26 gave a file identical row-for-row to the original pull.

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python scripts/fetch_statcast.py 2022 2023 2024 2025 2026   # ~2 min per season; pybaseball caches raw pulls
.venv/bin/python scripts/baseline_logit.py | tee results/baseline_logit.txt   # ~30 s
```

| Path | Contents |
|---|---|
| `scripts/fetch_statcast.py` | Downloads called pitches per season to `data/called_pitches_<year>.parquet` |
| `scripts/baseline_logit.py` | Models A and B, accuracy by edge distance, count effect |
| `results/baseline_logit.txt` | Output of the last run |
| `results/model_b.json` | Model B coefficients on the raw feature scale, plus its test metrics |
| `METHODOLOGY.md` | Data rules, evaluation protocol, extra checks, goals, changelog |
| `demo/index.html` | Static interactive strike zone running Model B |
| `MLB_Python_Code_Report.ipynb` | The original 2023 analysis (Kelley, Readey, Hasler) |
