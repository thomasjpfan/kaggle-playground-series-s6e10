# EDA: Playground Series S6E10 (`train.csv`)

The EDA runs on Modal with polars (`uv run modal run eda.py` calls `eda` in `eda.py`).
The data is read from the `kaggle` volume at `/data/playground-series-s6e10/`.

## Overview

- **Train:** 699,635 rows × 23 columns. **Test:** 299,844 rows × 22 columns. Test ids continue from train (699,635–999,478).
- **Domain:** a synthetic version of the classic *Airline Passenger Satisfaction* dataset.
- **Target:** `satisfaction` (boolean). The positive rate is **44.4%**, so classes are fairly balanced.
- **Submission:** a probability per `id`. The sample submission fills in the base rate of 0.4436, so the metric is probably ROC AUC or log loss.
- **Features:**
  - 4 categorical: `Gender`, `Customer Type`, `Type of Travel`, `Class`
  - 14 ordinal ratings on a 0–5 scale
  - `Age`, `Flight Distance`, and 2 delay columns
- **Data quality:**
  - The only missing values are in `Arrival Delay in Minutes`: 292 in train and 130 in test.
  - There are no duplicate rows (ignoring `id`).
- **Train/test drift:** essentially none. The means and standard deviations of every numeric feature, and the shares of every category, match to about the 3rd decimal place. A random K-fold CV should track the leaderboard well.

## Target vs. categorical features

| Feature | Level | Share | Satisfied rate |
|---|---|---|---|
| Customer Type | Loyal | 82.5% | 49.5% |
| | disloyal | 17.5% | 20.1% |
| Type of Travel | Business travel | 71.1% | 58.4% |
| | Personal Travel | 28.9% | **9.7%** |
| Class | Business | 48.9% | **72.5%** |
| | Eco | 46.8% | 16.8% |
| | Eco Plus | 4.3% | 24.2% |
| Gender | Female / Male | 50 / 50 | 43.8% / 44.9% |

- **Travel type and class** are the strongest categorical signals.
- **The two interact.** Personal travelers are about 10% satisfied whatever their class, including Business class (11%). Only business travelers in Business class reach 73%. A tree model will find this on its own; a linear model needs a `Type of Travel × Class` cross feature.
- **Gender** is nearly uninformative.

## Ratings (0–5)

Satisfied rate by rating value:

| Feature | 0 | 1 | 2 | 3 | 4 | 5 | % zeros |
|---|---|---|---|---|---|---|---|
| Inflight wifi service | **0.89** | 0.39 | 0.27 | 0.28 | 0.62 | 0.95 | 1.9% |
| Departure/Arrival time convenient | 0.48 | 0.51 | 0.45 | 0.45 | 0.40 | 0.44 | 3.7% |
| Ease of Online booking | **0.71** | 0.42 | 0.31 | 0.32 | 0.56 | 0.75 | 2.3% |
| Gate location | 0.54 | 0.51 | 0.47 | 0.34 | 0.41 | 0.63 | 0.2% |
| Food and drink | 0.22 | 0.17 | 0.39 | 0.41 | 0.53 | 0.57 | 0.1% |
| Online boarding | **0.62** | 0.11 | 0.09 | 0.13 | 0.64 | 0.88 | 1.3% |
| Seat comfort | 0.18 | 0.17 | 0.17 | 0.19 | 0.57 | 0.67 | 0.1% |
| Inflight entertainment | 0.14 | 0.13 | 0.17 | 0.24 | 0.62 | 0.67 | 0.2% |
| On-board service | 0.15 | 0.17 | 0.22 | 0.29 | 0.54 | 0.66 | 0.1% |
| Leg room service | 0.27 | 0.19 | 0.25 | 0.27 | 0.59 | 0.62 | 0.2% |
| Baggage handling | – | 0.31 | 0.27 | 0.21 | 0.48 | 0.61 | 0% |
| Checkin service | 0.18 | 0.20 | 0.22 | 0.46 | 0.47 | 0.61 | 0.1% |
| Cleanliness | 0.17 | 0.16 | 0.18 | 0.44 | 0.55 | 0.63 | 0.1% |

Observations:

- **`Online boarding` is the single best feature.** Its Pearson correlation with the target is 0.56 (Spearman 0.60), and the effect is close to a step function: ratings 1–3 give about 10% satisfied, 4 gives 64%, and 5 gives 88%.
- **Most ratings act like thresholds rather than linear scores.** The jump usually sits between 3 and 4 (seat comfort, entertainment, on-board service, leg room). For check-in and cleanliness it sits between 2 and 3. Treating ratings as categorical (one-hot or target encoding) should help linear models; trees handle it natively.
- **A 0 means "not applicable", not "worst".** This matches the original dataset, where 0 meant the question wasn't answered. For wifi, online booking and online boarding, a 0 is associated with *high* satisfaction (89%, 71% and 62%). Wifi is U-shaped: 0 and 5 are both very satisfied.
  - **Count of zeros per row:** rows with 3 or more zero ratings are 86–94% satisfied (about 9.8k rows), versus the 44% base rate. The `n_zeros` count, or per-column `is_zero` flags, is worth adding.
- **Weak ratings:** `Departure/Arrival time convenient` and `Gate location` carry almost no monotone signal (|r| < 0.05). There may still be some non-linear signal in Gate location (3 gives 34%, 5 gives 63%).
- **Ratings are strongly correlated with each other in two clusters:**
  - Comfort: seat comfort, entertainment, cleanliness and food all have r of about 0.65–0.76.
  - Digital and convenience: wifi and online booking have r = 0.77, and these also correlate with gate location and time convenience.

  Row aggregates should help. The mean rating alone has r = 0.51, and the count of 5s has r = 0.45.

## Numeric features

- **Flight Distance** has a strong positive effect (r = 0.37). Satisfaction is about 30% below 1,000 miles, 47% at 1,000–2,000, 71% at 2,000–3,000, and 80% above 3,000. This is likely confounded with Business class and business travel.
- **Age** has a non-monotonic effect, which a linear term will miss:

  | Age | ≤18 | 19–25 | 26–40 | 41–60 | >60 |
  |---|---|---|---|---|---|
  | Satisfied | 14% | 30% | 39% | **60%** | 14% |

- **Delays:**
  - Both columns are about 90% zeros and heavily right-skewed (max about 490 minutes).
  - The signal is weak but present: satisfaction drops from 45% with no arrival delay to 37% at 1–15 minutes and 31% above 15 minutes.
  - The two delay columns are almost the same feature (r = 0.84, exactly equal in 87% of rows).
  - The 292 missing arrival delays have a lower satisfied rate (36%). Keep a missing flag, or impute from departure delay.
  - Useful transforms: `log1p`, an `is_delayed` flag, and `arr - dep` (time made up or lost in the air).

## Modeling thoughts

1. **Baseline:** a GBDT (LightGBM, XGBoost or CatBoost) on the raw features, with the 4 categoricals as native categoricals, evaluated with stratified 5-fold CV and ROC AUC. With the clean, non-linear signal described above, this should already score very high (the original dataset reaches about 0.99 AUC).
2. **Features to try:**
   - zero/"N/A" flags and `n_zeros`
   - rating mean, min, max and count of 5s
   - the `Type of Travel × Class` cross
   - `arr - dep` delay
   - treating ratings as categorical (CatBoost)
3. **Ensembling:** with no train/test drift, plain CV is reliable, so ensembling several GBDTs plus a categorical-ratings logistic regression is a reasonable next step.

## Follow-up EDA and experiments

These were run with `eda2.py`:
- `uv run modal run eda2.py` runs the data checks and the default feature sets.
- `--no-checks --kinds base,te_fd` runs only the chosen feature sets.

Every number below is LightGBM out-of-fold (OOF) AUC on the same 5 folds as `eval.py`.

### Feature-set results

| Feature set | OOF AUC | Δ vs base |
|---|---|---|
| base (raw features, native categoricals) | 0.95883 | — |
| + `n_zeros`, `n_fives`, `n_high`, rating mean and min, travel×class, `arr - dep` | 0.95869 | −0.0001 |
| + the same, with ratings as categoricals | 0.95868 | −0.0002 |
| + count encoding of `Age` and `Flight Distance` | 0.95952 | +0.0007 |
| + count encoding of all 4 numeric columns | 0.95950 | +0.0007 |
| + counts, with `Age` as a categorical | 0.95898 | +0.0002 |
| **+ counts + in-fold target encoding of `Flight Distance`** | **0.96022** | **+0.0014** |
| + target encoding of all numeric columns | 0.96013 | +0.0013 |
| + target encoding of numeric columns and of `FD×Class`, `FD×Travel`, `Age×Customer Type` | 0.96012 | +0.0013 |

The best variant is now built into `eval.py`, which scores 0.96025 OOF.

- **The engineered features from the first EDA pass don't help.** That includes the zero counts, rating aggregates and the travel × class cross. Trees already find these patterns. Drop this line of work for GBDTs.
- **Exact `Flight Distance` values carry signal beyond the smooth trend.** For values with at least 30 rows, each value's satisfied rate differs from a rolling mean over its 21 neighbours about 19× more than binomial noise would explain. The likely cause is that the generator reuses distances from specific routes in the original data. Count and target encoding capture this. Target-encoding the other numeric columns adds nothing.

### Data checks

- **Duplicates:** there are no exact feature duplicates within train, and no test row matches a train row exactly.
- **Value coverage:** `Age` has 75 unique values and `Flight Distance` has 3,474. Only 61 test rows have a `Flight Distance` value not seen in train.
- **Label noise is the ceiling.** The OOF predictions are well calibrated decile by decile. Even so, 3.7% of rows are confidently wrong (|pred − y| > 0.9). An example is Business travel in Business class with `Online boarding = 5` labeled unsatisfied. The original dataset reaches about 0.99 AUC, but here gains are likely to come in steps of about 0.001.

### Where the model is weak (OOF AUC within segment)

| Segment | Rows | AUC within segment |
|---|---|---|
| Business travel | 497k | 0.955 |
| Personal travel | 202k | **0.828** |
| Disloyal customers | 123k | 0.923 |
| Rows with `n_zeros = 4` | 2.9k | 0.54 |

- **Personal travel:** only wifi (r = 0.24), online booking (0.22) and online boarding (0.16) matter. The comfort ratings have near-zero correlation in this segment. Wifi alone splits it: ratings of 1–3 are about 4% satisfied, 4 is 24%, and 0 or 5 is 79–91%.
- **`Customer Type` × `Age`:** loyal 41–60 year-olds are 62% satisfied. Disloyal customers are 14–28% satisfied in every age band.

## Models, tuning and blending

The competition uses only its own data; the original dataset is not used. Every score is OOF AUC on the shared 5 folds.

| Model | OOF AUC |
|---|---|
| LightGBM, raw features | 0.95883 |
| LightGBM + count and `Flight Distance` target encoding | 0.96023 |
| **LightGBM, tuned** (`eval.py`) | **0.96053** |
| CatBoost `native` (4 string columns as categoricals) | 0.95838 |
| CatBoost `allcat` (also ratings, `Age` and `Flight Distance` as categoricals) | 0.96062 |
| **CatBoost `allcat`, tuned** (`eval_catboost.py`) | **0.96075** |
| SDM `tabiclv2`, 50k context | 0.95749 |
| SDM `kumo-tabular`, 50k / 200k context | 0.95807 / 0.95895 |
| **Rank blend: 0.6 CatBoost + 0.4 LightGBM (tuned)** | **≈ 0.9610** |

- **CatBoost needs exact values as categoricals.** `allcat` beats `native` by +0.0022. Tuning (`tune.py`, 60 trials on fold 0) settled on `depth=7`, `max_ctr_complexity=4`, `one_hot_max_size=2` and light L2. `max_ctr_complexity=1` was always worst, so combinations of categoricals carry signal.
- **LightGBM needs fine bins.** All of the top 10 tuned configs use `max_bin=4095`; the default 255 bins merge the exact `Flight Distance` values. Stronger L2 (about 15), `feature_fraction` about 0.5, and low `cat_smooth` also helped.
- **Noise floor.** GPU CatBoost varies by about ±0.0002 between identical runs, which is as large as the gaps between the top tuning configs.
- **Blending.** The two trees correlate at 0.97 in rank. The SDM models are more different (0.94) but too weak to earn weight, even at a 200k context. An equal average of all models scores below CatBoost alone.

### Next ideas

1. **Check CV against the leaderboard** with the first submission (`blend.py --submit-weights`).
2. **If they agree:** try a lower learning rate (0.02) for both trees, or a full-fold SDM context for diversity.
