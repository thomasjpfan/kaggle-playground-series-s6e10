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
3. **Original data:** this competition is generated from the public *Airline Passenger Satisfaction* dataset (about 130k rows). Appending the original data to training, or adding an "is_original" indicator, is a common win in Playground series.
4. **Ensembling:** with no train/test drift, plain CV is reliable, so ensembling several GBDTs plus a categorical-ratings logistic regression is a reasonable next step.
