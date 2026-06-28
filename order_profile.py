"""
order_probs.py
==============
Customer 14-Day Purchase Profile Prediction
UNORG B2B Demand Forecasting Pipeline — Part 1

Outputs:
  - Console: training milestones and evaluation metrics
  - Optional CSV: customer_id | predicted_order_count | day_1 ... day_14
"""

# ==============================================================
# STEP 0: VERIFY REQUIREMENTS
# ==============================================================
import sys
import importlib.util

REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "catboost": "catboost",
    "sklearn": "scikit-learn",
}

missing = []
for module, package in REQUIRED.items():
    if importlib.util.find_spec(module) is None:
        missing.append(package)

if missing:
    print("\n[ERROR] The following required packages are not installed:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\nInstall them with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

print("All required packages verified.")

# ==============================================================
# STEP 1: IMPORTS
# ==============================================================
import os
from collections import deque

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================
# STEP 2: GPU CHECK
# ==============================================================
try:
    import torch
    TASK_TYPE = "GPU" if torch.cuda.is_available() else "CPU"
except ImportError:
    # torch not installed — let CatBoost decide
    # CatBoost will fall back to CPU if no GPU is found
    TASK_TYPE = "CPU"

print(f"Using device: {TASK_TYPE}")

# ==============================================================
# STEP 3: FILE PATH
# ==============================================================
DEFAULT_FILE = "order_data_last_six_month.csv"

user_input = input(
    f"\nEnter path to order data CSV [press Enter to use default: '{DEFAULT_FILE}']: "
).strip()

file_path = user_input if user_input else DEFAULT_FILE

if not os.path.exists(file_path):
    print(f"\n[ERROR] File not found: '{file_path}'")
    print("Please verify the path and try again.")
    sys.exit(1)

print(f"Loading data from: {file_path}")

# ==============================================================
# STEP 4: READ & SORT DATA
# ==============================================================
df = pd.read_csv(file_path)
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)

# Store the true max date of the original dataset.
# This is treated as "today" — the 14-day profile begins after this date.
TRUE_MAX_DATE = df["order_date"].max()
print(f"\nDataset max date (treated as 'today'): {TRUE_MAX_DATE.date()}")
print(
    f"Purchase profile will cover: "
    f"{(TRUE_MAX_DATE + pd.Timedelta(days=1)).date()} "
    f"to {(TRUE_MAX_DATE + pd.Timedelta(days=14)).date()}"
)

print("\nData loaded and sorted.")

# ==============================================================
# STEP 5: AGGREGATE TO CUSTOMER-DAY LEVEL
# ==============================================================
daily_orders = (
    df.groupby(["customer_id", "order_date"])
    .agg(
        num_orders=("order_id", "count"),
        net_order_amount=("net_order_amount", "sum"),
    )
    .reset_index()
    .rename(columns={"order_date": "date"})
)
daily_orders["order_placed"] = 1

print("Customer-day aggregation completed.")

# ==============================================================
# STEP 6: CREATE CUSTOMER × DATE PANEL
# ==============================================================
all_customers = daily_orders["customer_id"].unique()
all_dates = pd.date_range(daily_orders["date"].min(), daily_orders["date"].max(), freq="D")

panel = pd.MultiIndex.from_product(
    [all_customers, all_dates], names=["customer_id", "date"]
).to_frame(index=False)

df_daily = panel.merge(daily_orders, on=["customer_id", "date"], how="left")
df_daily["order_placed"] = df_daily["order_placed"].fillna(0).astype(int)

print("Customer × date panel created.")

# ==============================================================
# STEP 7: CALENDAR FEATURES
# ==============================================================
df_daily["day_of_week"] = df_daily["date"].dt.dayofweek
df_daily["month"] = df_daily["date"].dt.month

# ==============================================================
# STEP 8: FREQUENCY FEATURES (past-only, shift(1) prevents leakage)
# ==============================================================
for window in [7, 14, 30]:
    df_daily[f"frequency_{window}d"] = (
        df_daily.groupby("customer_id")["order_placed"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
    )

df_daily["freq7_freq14_ratio"] = df_daily["frequency_7d"] / df_daily["frequency_14d"]
df_daily["freq7_freq30_ratio"] = df_daily["frequency_7d"] / df_daily["frequency_30d"]

# ==============================================================
# STEP 9: DAYS SINCE FIRST ORDER
# ==============================================================
first_order_date = df.groupby("customer_id")["order_date"].min()
df_daily["days_since_first_order"] = (
    df_daily["date"] - df_daily["customer_id"].map(first_order_date)
).dt.days

print("Base feature engineering completed.")

# ==============================================================
# STEP 10: FUTURE 14-DAY PROFILE TARGETS
# (built on full dataset — including last 14 days)
# ==============================================================
for i in range(1, 15):
    df_daily[f"order_day_{i}"] = (
        df_daily.groupby("customer_id")["order_placed"].shift(-i)
    )

target_cols = [f"order_day_{i}" for i in range(1, 15)]
df_daily["target_order_count"] = df_daily[target_cols].sum(axis=1)

# ==============================================================
# STEP 11: SEQUENTIAL FEATURES (recency, amounts, peak day)
# Computed on full panel — leakage-safe (state updated AFTER read)
# ==============================================================
seq_cols = [
    "days_since_last_1_order", "days_since_last_2_order", "days_since_last_3_order",
    "last_1_net_order_amount", "last_2_net_order_amount", "last_3_net_order_amount",
    "mean_net_order_amount_till_now", "total_orders",
    "peak_order_day_till_now", "remaining_days_till_next_peak_day",
]
for col in seq_cols:
    df_daily[col] = np.nan

print("Computing sequential features per customer (this may take a moment)...")

for customer_id, idx in df_daily.groupby("customer_id").groups.items():
    idx = list(idx)
    order_dates = deque(maxlen=3)
    order_amounts = deque(maxlen=3)
    total_orders = 0
    amount_sum = 0.0
    weekday_counts = np.zeros(7, dtype=int)
    weekday_last_seen = {i: pd.Timestamp.min for i in range(7)}

    for row_idx in idx:
        row = df_daily.loc[row_idx]
        current_date = row["date"]
        current_weekday = row["day_of_week"]

        # Recency
        if len(order_dates) >= 1:
            df_daily.at[row_idx, "days_since_last_1_order"] = (current_date - order_dates[-1]).days
        if len(order_dates) >= 2:
            df_daily.at[row_idx, "days_since_last_2_order"] = (current_date - order_dates[-2]).days
        if len(order_dates) >= 3:
            df_daily.at[row_idx, "days_since_last_3_order"] = (current_date - order_dates[-3]).days

        # Amounts
        if len(order_amounts) >= 1:
            df_daily.at[row_idx, "last_1_net_order_amount"] = order_amounts[-1]
        if len(order_amounts) >= 2:
            df_daily.at[row_idx, "last_2_net_order_amount"] = order_amounts[-2]
        if len(order_amounts) >= 3:
            df_daily.at[row_idx, "last_3_net_order_amount"] = order_amounts[-3]

        # Running mean
        if total_orders > 0:
            df_daily.at[row_idx, "mean_net_order_amount_till_now"] = amount_sum / total_orders

        # Total orders
        df_daily.at[row_idx, "total_orders"] = total_orders

        # Peak weekday
        if total_orders > 0:
            max_count = weekday_counts.max()
            candidate_days = np.where(weekday_counts == max_count)[0]
            peak_day = max(candidate_days, key=lambda d: weekday_last_seen[d])
            df_daily.at[row_idx, "peak_order_day_till_now"] = peak_day
            df_daily.at[row_idx, "remaining_days_till_next_peak_day"] = (peak_day - current_weekday) % 7

        # Update state AFTER reading (leakage-safe)
        if row["order_placed"] == 1:
            order_dates.append(current_date)
            order_amounts.append(row["net_order_amount"])
            total_orders += 1
            amount_sum += row["net_order_amount"]
            weekday_counts[current_weekday] += 1
            weekday_last_seen[current_weekday] = current_date

print("Sequential features completed.")

# ==============================================================
# STEP 12: IDENTIFY INFERENCE ROWS (last 14 days of full dataset)
# These rows have full features but no valid targets — used for
# final prediction output, not for training or evaluation.
# ==============================================================
inference_df = df_daily[df_daily["date"] > (TRUE_MAX_DATE - pd.Timedelta(days=14))].copy()

# ==============================================================
# STEP 13: REMOVE LAST 14 DAYS FROM TRAINING DATA
# Rows where target window extends beyond dataset are unreliable.
# ==============================================================
max_date = df_daily["date"].max()
df_daily = df_daily[
    df_daily["date"] <= max_date - pd.Timedelta(days=14)
].reset_index(drop=True)

print(f"Training panel size after removing last 14 days: {df_daily.shape[0]:,} rows")

# ==============================================================
# STEP 14: TRAIN / TEST SPLIT (85 / 15 chronological)
# ==============================================================
FEATURE_COLS = [
    "frequency_7d", "frequency_14d", "frequency_30d",
    "freq7_freq14_ratio", "freq7_freq30_ratio",
    "days_since_last_1_order", "days_since_last_2_order", "days_since_last_3_order",
    "days_since_first_order", "total_orders",
    "last_1_net_order_amount", "last_2_net_order_amount", "last_3_net_order_amount",
    "mean_net_order_amount_till_now",
    "peak_order_day_till_now", "remaining_days_till_next_peak_day",
    "day_of_week", "month",
]

dates = np.sort(df_daily["date"].unique())
n_dates = len(dates)
train_end = int(0.85 * n_dates)

train_dates = dates[:train_end]
test_dates = dates[train_end:]

train_df = df_daily[df_daily["date"].isin(train_dates)].reset_index(drop=True)
test_df = df_daily[df_daily["date"].isin(test_dates)].reset_index(drop=True)

# Validation split from within train (last 15% of train for early stopping)
val_end_idx = int(0.85 * len(np.sort(train_df["date"].unique())))
val_dates_inner = np.sort(train_df["date"].unique())[val_end_idx:]
val_df = train_df[train_df["date"].isin(val_dates_inner)].reset_index(drop=True)
train_df_final = train_df[~train_df["date"].isin(val_dates_inner)].reset_index(drop=True)

X_train = train_df_final[FEATURE_COLS]
y_train = train_df_final["target_order_count"]
X_val = val_df[FEATURE_COLS]
y_val = val_df["target_order_count"]
X_test = test_df[FEATURE_COLS]
y_test = test_df["target_order_count"]

print(f"Train rows: {len(X_train):,} | Val rows: {len(X_val):,} | Test rows: {len(X_test):,}")

# ==============================================================
# STEP 15: COUNT MODEL — OOF PREDICTIONS ON TRAIN
# ==============================================================
print("\nStarting Count Regressor training (OOF folds)...")

train_count_pred = np.zeros(len(X_train))
tscv = TimeSeriesSplit(n_splits=5)

for fold, (tr_idx, oof_idx) in enumerate(tscv.split(X_train), start=1):
    fold_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        task_type=TASK_TYPE,
        random_seed=42,
        verbose=0,
    )
    fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    train_count_pred[oof_idx] = fold_model.predict(X_train.iloc[oof_idx])
    print(f"  Fold {fold}/5 completed.")

# ==============================================================
# STEP 16: COUNT MODEL — FINAL MODEL
# ==============================================================
print("\nTraining final Count Regressor...")

count_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    task_type=TASK_TYPE,
    random_seed=42,
    verbose=0,
)
count_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
)

print("Count Regressor training completed.")

# ==============================================================
# STEP 17: COUNT MODEL — PREDICTIONS & METRICS
# ==============================================================
val_count_pred = count_model.predict(X_val)
test_count_pred = count_model.predict(X_test)

train_count_pred_int = np.round(train_count_pred).clip(0, 14).astype(int)
test_count_pred_int = np.round(test_count_pred).clip(0, 14).astype(int)
val_count_pred_int = np.round(val_count_pred).clip(0, 14).astype(int)

count_mae = mean_absolute_error(y_test, test_count_pred)
count_rmse = np.sqrt(mean_squared_error(y_test, test_count_pred))
count_exact_acc = (test_count_pred_int == y_test.values).mean()
count_tol1_acc = (np.abs(test_count_pred_int - y_test.values) <= 1).mean()

print("\n--- Count Regressor | Test Set Metrics ---")
print(f"  MAE               : {count_mae:.4f}")
print(f"  RMSE              : {count_rmse:.4f}")
print(f"  Exact Accuracy    : {count_exact_acc:.4f}")
print(f"  ±1 Accuracy       : {count_tol1_acc:.4f}")

train_df_final["predicted_order_count"] = train_count_pred
val_df["predicted_order_count"] = val_count_pred
test_df["predicted_order_count"] = test_count_pred

# ==============================================================
# STEP 18: PROFILE MODEL — BUDGET-AWARE SEQUENTIAL CHAIN
# ==============================================================
TARGET_COLS = [f"order_day_{i}" for i in range(1, 15)]

X_train_base = train_df_final[FEATURE_COLS].fillna(-1).reset_index(drop=True)
Y_train = train_df_final[TARGET_COLS].reset_index(drop=True)

X_val_base = val_df[FEATURE_COLS].fillna(-1).reset_index(drop=True)
Y_val = val_df[TARGET_COLS].reset_index(drop=True)

X_test_base = test_df[FEATURE_COLS].fillna(-1).reset_index(drop=True)
Y_test = test_df[TARGET_COLS].reset_index(drop=True)

print("\nStarting Budget-Aware Sequential Chain training (14 models)...")

chain_models = []

for day in range(14):
    X_day = X_train_base.copy()

    # Feed prior TRUE labels as features (teacher forcing during train)
    for prev in range(day):
        X_day[f"prev_day_{prev + 1}"] = Y_train.iloc[:, prev].values

    # Remaining budget = predicted count minus orders already assigned
    budget = train_count_pred - Y_train.iloc[:, :day].sum(axis=1).values
    X_day["remaining_budget"] = budget

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        task_type=TASK_TYPE,
        random_seed=42,
        verbose=0,
    )
    model.fit(X_day, Y_train.iloc[:, day])
    chain_models.append(model)
    print(f"  Day {day + 1}/14 model trained.")

print("Sequential chain training completed.")

# ==============================================================
# STEP 19: THRESHOLD OPTIMIZATION ON VALIDATION SET
# ==============================================================
print("\nOptimizing decision threshold on validation set...")

def tolerant_jaccard(y_true, y_pred, tolerance=1):
    true_days = np.where(y_true == 1)[0]
    pred_days = np.where(y_pred == 1)[0]
    matched_true = set()
    intersection = 0
    for p in pred_days:
        for j, t in enumerate(true_days):
            if j in matched_true:
                continue
            if abs(p - t) <= tolerance:
                matched_true.add(j)
                intersection += 1
                break
    union = len(true_days) + len(pred_days) - intersection
    return 1.0 if union == 0 else intersection / union

val_probs = np.zeros((len(X_val_base), 14))
val_preds_tmp = np.zeros((len(X_val_base), 14), dtype=int)

for day in range(14):
    X_day = X_val_base.copy()
    for prev in range(day):
        X_day[f"prev_day_{prev + 1}"] = val_preds_tmp[:, prev]
    budget = val_count_pred_int - val_preds_tmp[:, :day].sum(axis=1)
    X_day["remaining_budget"] = budget
    probs = chain_models[day].predict_proba(X_day)[:, 1]
    val_probs[:, day] = probs

threshold_grid = np.arange(0.05, 0.55, 0.02)
best_threshold = 0.5
best_val_score = -1

for t in threshold_grid:
    val_pred_tmp = (val_probs >= t).astype(int)
    score = np.mean([
        tolerant_jaccard(Y_val.iloc[i].values, val_pred_tmp[i])
        for i in range(len(Y_val))
    ])
    if score > best_val_score:
        best_val_score = score
        best_threshold = t

print(f"  Best threshold    : {best_threshold:.2f}")
print(f"  Val Tolerant Jaccard: {best_val_score:.4f}")

# ==============================================================
# STEP 20: TEST SET EVALUATION
# ==============================================================
print("\nRunning inference on test set...")

Y_pred = np.zeros((len(X_test_base), 14), dtype=int)

for day in range(14):
    X_day = X_test_base.copy()
    for prev in range(day):
        X_day[f"prev_day_{prev + 1}"] = Y_pred[:, prev]
    budget = test_count_pred_int - Y_pred[:, :day].sum(axis=1)
    X_day["remaining_budget"] = budget
    probs = chain_models[day].predict_proba(X_day)[:, 1]
    Y_pred[:, day] = (probs >= best_threshold).astype(int)

micro_f1 = f1_score(Y_test, Y_pred, average="micro")
ham_loss = hamming_loss(Y_test, Y_pred)
count_error = np.mean(np.abs(Y_pred.sum(axis=1) - Y_test.values.sum(axis=1)))

strict_scores = [
    jaccard_score(Y_test.iloc[i], Y_pred[i], zero_division=0)
    for i in range(len(Y_test))
]
tol_scores = [
    tolerant_jaccard(Y_test.iloc[i].values, Y_pred[i])
    for i in range(len(Y_test))
]

print("\n--- Budget-Aware Sequential Chain | Test Set Metrics ---")
print(f"  Micro F1              : {micro_f1:.4f}")
print(f"  Hamming Loss          : {ham_loss:.4f}")
print(f"  Mean Count Error      : {count_error:.4f}")
print(f"  Strict Jaccard        : {np.mean(strict_scores):.4f}")
print(f"  Tolerant Jaccard      : {np.mean(tol_scores):.4f}")

# ==============================================================
# STEP 21: BUILD FINAL INFERENCE OUTPUT
# Treat TRUE_MAX_DATE as today. Run inference on the last 14-day
# window rows (which have complete features but no future targets).
# ==============================================================
print("\nBuilding 14-day purchase profile for all customers...")

# Use the last available row per customer from the full panel
# (the row at TRUE_MAX_DATE, i.e. the most recent feature state)
last_rows = (
    inference_df.sort_values("date")
    .groupby("customer_id")
    .last()
    .reset_index()
)

# Get count predictions for these inference rows
X_infer = last_rows[FEATURE_COLS].fillna(-1)
infer_count_pred = count_model.predict(X_infer)
infer_count_pred_int = np.round(infer_count_pred).clip(0, 14).astype(int)

# Run sequential chain on inference rows
X_infer_base = X_infer.reset_index(drop=True)
Y_infer = np.zeros((len(X_infer_base), 14), dtype=int)

for day in range(14):
    X_day = X_infer_base.copy()
    for prev in range(day):
        X_day[f"prev_day_{prev + 1}"] = Y_infer[:, prev]
    budget = infer_count_pred_int - Y_infer[:, :day].sum(axis=1)
    X_day["remaining_budget"] = budget
    probs = chain_models[day].predict_proba(X_day)[:, 1]
    Y_infer[:, day] = (probs >= best_threshold).astype(int)

# Assemble output dataframe
output_df = pd.DataFrame()
output_df["customer_id"] = last_rows["customer_id"].values

for i in range(1, 15):
    col_date = (TRUE_MAX_DATE + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
    output_df[col_date] = Y_infer[:, i - 1]

print(f"\n14-day purchase profile built for {len(output_df):,} customers.")
print(f"Profile window: day_1 = {(TRUE_MAX_DATE + pd.Timedelta(days=1)).date()} "
      f"... day_14 = {(TRUE_MAX_DATE + pd.Timedelta(days=14)).date()}")



# ==============================================================
# STEP 22: SAVE OUTPUT
# ==============================================================
save_choice = input("\nSave output to CSV? (1 = Yes, 0 = No): ").strip()

if save_choice == "1":
    out_path = input("Enter output file path [press Enter for 'customer_14day_profile.csv']: ").strip()
    out_path = out_path if out_path else "customer_14day_profile.csv"
    output_df.to_csv(out_path, index=False)
    print(f"Output saved to: {out_path}")
else:
    print("Output not saved.")

print("\nDone.")
