"""
inventory.py
============
Customer-Item Demand Prediction & Warehouse Inventory Aggregation
UNORG B2B Demand Forecasting Pipeline — Part 2

Outputs:
  - Console: training milestones and evaluation metrics
  - Optional CSV 1: customer_id | item_name | warehouse_id | predicted_qty_next14d
  - Optional CSV 2: warehouse_id | item_name | predicted_qty_next14d  (aggregated)
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
from collections import defaultdict, deque, Counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==============================================================
# STEP 2: GPU CHECK
# ==============================================================
try:
    import torch
    TASK_TYPE = "GPU" if torch.cuda.is_available() else "CPU"
except ImportError:
    TASK_TYPE = "CPU"

print(f"Using device: {TASK_TYPE}")

# ==============================================================
# STEP 3: FILE PATHS
# ==============================================================
DEFAULT_ORDERS_FILE = "order_data_last_six_month.csv"
DEFAULT_ITEMS_FILE = "associated_order_item_data_last_six_month.csv"

orders_input = input(
    f"\nEnter path to orders CSV [press Enter for default: '{DEFAULT_ORDERS_FILE}']: "
).strip()
items_input = input(
    f"Enter path to order items CSV [press Enter for default: '{DEFAULT_ITEMS_FILE}']: "
).strip()

orders_path = orders_input if orders_input else DEFAULT_ORDERS_FILE
items_path = items_input if items_input else DEFAULT_ITEMS_FILE

for path in [orders_path, items_path]:
    if not os.path.exists(path):
        print(f"\n[ERROR] File not found: '{path}'")
        print("Please verify the path and try again.")
        sys.exit(1)

print(f"\nOrders file : {orders_path}")
print(f"Items file  : {items_path}")

# ==============================================================
# STEP 4: LOAD DATA
# ==============================================================
print("\nLoading data...")

orders = pd.read_csv(
    orders_path,
    usecols=["order_date", "order_id", "customer_id", "warehouse_id"],
    dtype={"order_id": "int32", "customer_id": "int32", "warehouse_id": "int16"},
)

items = pd.read_csv(
    items_path,
    usecols=["order_id", "item_name", "quantity"],
    dtype={"order_id": "int32", "item_name": "category", "quantity": "float32"},
)

orders["order_date"] = pd.to_datetime(orders["order_date"], format="%d/%m/%Y").dt.normalize()

# Store the true max date of the original dataset — treated as "today"
TRUE_MAX_DATE = orders["order_date"].max()
print(f"Dataset max date (treated as 'today'): {TRUE_MAX_DATE.date()}")
print(
    f"Prediction window: "
    f"{(TRUE_MAX_DATE + pd.Timedelta(days=1)).date()} "
    f"to {(TRUE_MAX_DATE + pd.Timedelta(days=14)).date()}"
)

print("Data loaded.")

# ==============================================================
# STEP 5: BUILD DAY-LEVEL MASTER DATASET
# ==============================================================
print("\nBuilding day-level master dataset...")

master_df = orders.merge(items, on="order_id", how="inner", sort=False)

# Last warehouse per customer-item-day
last_wh = (
    master_df
    .sort_values(["customer_id", "order_date", "order_id"])
    .groupby(["customer_id", "order_date", "item_name"], observed=True, sort=False)["warehouse_id"]
    .last()
)

# Sum quantities per customer-item-day
daily_qty = (
    master_df
    .groupby(["customer_id", "order_date", "item_name"], observed=True, sort=False)["quantity"]
    .sum()
)

master_df = pd.concat([daily_qty, last_wh], axis=1).reset_index()

master_df["customer_id"] = master_df["customer_id"].astype("int32")
master_df["warehouse_id"] = master_df["warehouse_id"].astype("int16")
master_df["quantity"] = master_df["quantity"].astype("float32")
master_df["item_name"] = master_df["item_name"].astype("category")

master_df = master_df.sort_values(
    ["customer_id", "order_date", "item_name"], kind="mergesort"
).reset_index(drop=True)

print("Day-level master dataset built.")

# ==============================================================
# STEP 6: STREAMING FEATURE GENERATION
# ==============================================================
print("\nGenerating historical features per customer-item (streaming)...")
print("  This may take a few minutes depending on dataset size...")

WINDOW_SPECS = (
    ("q7", "sum7", 7),
    ("q14", "sum14", 14),
    ("q30", "sum30", 30),
)

def _new_item_state():
    return {
        "first_purchase_date": None,
        "last_purchase_date": None,
        "last_quantity": np.nan,
        "total_quantity": 0.0,
        "purchase_day_count": 0,
        "max_quantity": 0.0,
        "gap_sum": 0.0,
        "gap_count": 0,
        "last_purchase_order_day_count": None,
        "last_warehouse": np.nan,
        "warehouse_counts": defaultdict(int),
        "mode_warehouse": None,
        "mode_count": 0,
        "q7": deque(), "q14": deque(), "q30": deque(),
        "sum7": 0.0, "sum14": 0.0, "sum30": 0.0,
    }

def _purge_windows(st, current_ord):
    for dq_key, sum_key, window in WINDOW_SPECS:
        dq = st[dq_key]
        lower_bound = current_ord - window
        while dq and dq[0][0] < lower_bound:
            st[sum_key] -= dq.popleft()[1]

# Feature generation runs up to TRUE_MAX_DATE (full dataset).
# The last 14 days of rows will be used for inference only.
# Rows up to TRUE_MAX_DATE - 14 days are used for training/evaluation.
last_training_date = TRUE_MAX_DATE - pd.Timedelta(days=14)

customer_groups = {
    cid: df.sort_values(["order_date", "item_name"], kind="mergesort").reset_index(drop=True)
    for cid, df in master_df.groupby("customer_id", sort=False)
}

customer_start_date = (
    master_df.groupby("customer_id", sort=False)["order_date"]
    .min().dt.normalize().to_dict()
)

model_parts = []

for customer_id, cust_df in customer_groups.items():
    start_date = pd.Timestamp(customer_start_date[customer_id]).normalize()
    if start_date > TRUE_MAX_DATE:
        continue

    cust_lookup = defaultdict(list)
    for row in cust_df.itertuples(index=False):
        cust_lookup[row.order_date].append((row.item_name, float(row.quantity), int(row.warehouse_id)))

    item_state = {}
    active_items = []
    active_set = set()
    rows = []
    order_day_count = 0

    current_date = start_date
    while current_date <= TRUE_MAX_DATE:
        current_ord = current_date.toordinal()

        for item_name in active_items:
            _purge_windows(item_state[item_name], current_ord)

        for item_name in active_items:
            st = item_state[item_name]
            if st["first_purchase_date"] is None or st["last_purchase_date"] is None:
                continue

            days_since_last = (current_date - st["last_purchase_date"]).days
            days_since_first = (current_date - st["first_purchase_date"]).days
            mean_qty = st["total_quantity"] / st["purchase_day_count"] if st["purchase_day_count"] > 0 else np.nan
            avg_gap = st["gap_sum"] / st["gap_count"] if st["gap_count"] > 0 else np.nan
            stop_ratio = days_since_last / avg_gap if pd.notna(avg_gap) and avg_gap > 0 else np.nan
            orders_since_last = (
                order_day_count - st["last_purchase_order_day_count"]
                if st["last_purchase_order_day_count"] is not None else np.nan
            )

            rows.append({
                "customer_id": customer_id,
                "item_name": item_name,
                "current_date": current_date,
                "purchase_day_count_of_this_item": st["purchase_day_count"],
                "days_since_last_order_of_this_item": float(days_since_last),
                "days_since_first_order_of_this_item": float(days_since_first),
                "last_quantity_of_this_item": float(st["last_quantity"]),
                "mean_quantity_of_this_item_till_now": float(mean_qty) if pd.notna(mean_qty) else np.nan,
                "total_quantity_of_this_item_till_now": float(st["total_quantity"]),
                "item_quantity_last7d": float(st["sum7"]),
                "item_quantity_last14d": float(st["sum14"]),
                "item_quantity_last30d": float(st["sum30"]),
                "max_quantity_of_this_item": float(st["max_quantity"]),
                "avg_days_between_item_orders": float(avg_gap) if pd.notna(avg_gap) else np.nan,
                "stop_ratio": float(stop_ratio) if pd.notna(stop_ratio) else np.nan,
                "orders_since_last_purchase_of_item": float(orders_since_last) if pd.notna(orders_since_last) else np.nan,
                "general_warehouse_of_this_item_till_now": st["mode_warehouse"],
                "last_warehouse_of_this_item": st["last_warehouse"],
            })

        todays_orders = cust_lookup.get(current_date, [])
        if todays_orders:
            order_day_count += 1
            for item_name, qty, warehouse_id in todays_orders:
                if item_name not in item_state:
                    item_state[item_name] = _new_item_state()
                st = item_state[item_name]
                qty = float(qty)
                if st["purchase_day_count"] == 0:
                    st["first_purchase_date"] = current_date
                    if item_name not in active_set:
                        active_set.add(item_name)
                        active_items.append(item_name)
                else:
                    gap = (current_date - st["last_purchase_date"]).days
                    st["gap_sum"] += float(gap)
                    st["gap_count"] += 1
                st["last_purchase_date"] = current_date
                st["last_quantity"] = qty
                st["total_quantity"] += qty
                st["purchase_day_count"] += 1
                if qty > st["max_quantity"]:
                    st["max_quantity"] = qty
                st["last_purchase_order_day_count"] = order_day_count
                st["last_warehouse"] = warehouse_id
                st["warehouse_counts"][warehouse_id] += 1
                wh_count = st["warehouse_counts"][warehouse_id]
                if st["mode_warehouse"] is None or wh_count >= st["mode_count"]:
                    st["mode_warehouse"] = warehouse_id
                    st["mode_count"] = wh_count
                st["q7"].append((current_ord, qty)); st["sum7"] += qty
                st["q14"].append((current_ord, qty)); st["sum14"] += qty
                st["q30"].append((current_ord, qty)); st["sum30"] += qty

        current_date += pd.Timedelta(days=1)

    if rows:
        model_parts.append(pd.DataFrame.from_records(rows))

model_df = pd.concat(model_parts, ignore_index=True) if model_parts else pd.DataFrame()

if not model_df.empty:
    model_df["customer_id"] = model_df["customer_id"].astype("int32")
    model_df["current_date"] = pd.to_datetime(model_df["current_date"]).dt.normalize()
    float_cols = [
        "days_since_last_order_of_this_item", "days_since_first_order_of_this_item",
        "last_quantity_of_this_item", "mean_quantity_of_this_item_till_now",
        "total_quantity_of_this_item_till_now", "item_quantity_last7d",
        "item_quantity_last14d", "item_quantity_last30d", "max_quantity_of_this_item",
        "avg_days_between_item_orders", "stop_ratio", "orders_since_last_purchase_of_item",
    ]
    for col in float_cols:
        model_df[col] = model_df[col].astype("float32")
    model_df["purchase_day_count_of_this_item"] = model_df["purchase_day_count_of_this_item"].astype("int16")
    for col in ["item_name", "general_warehouse_of_this_item_till_now", "last_warehouse_of_this_item"]:
        model_df[col] = model_df[col].astype("category")
    model_df = model_df.sort_values(
        ["customer_id", "current_date", "item_name"], kind="mergesort"
    ).reset_index(drop=True)

print("Historical feature generation completed.")

# ==============================================================
# STEP 7: GENERATE FUTURE TARGETS
# ==============================================================
print("\nGenerating future 14-day quantity targets...")

def _mode_latest(values):
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return np.nan
    counts = Counter(vals)
    max_count = max(counts.values())
    tied = {v for v, c in counts.items() if c == max_count}
    for v in reversed(vals):
        if v in tied:
            return v
    return np.nan

qty_col = "daily_quantity" if "daily_quantity" in master_df.columns else "quantity"

daily_purchase_df = (
    master_df.groupby(
        ["customer_id", "item_name", "order_date"],
        observed=True, sort=False, as_index=False
    ).agg(daily_quantity=(qty_col, "sum"), warehouse_id=("warehouse_id", "last"))
)
daily_purchase_df = daily_purchase_df.sort_values(
    ["customer_id", "item_name", "order_date"], kind="mergesort"
).reset_index(drop=True)

dataset_end_date = master_df["order_date"].max()
target_parts = []

for (customer_id, item_name), grp in daily_purchase_df.groupby(
    ["customer_id", "item_name"], observed=True, sort=False
):
    grp = grp.sort_values("order_date", kind="mergesort").reset_index(drop=True)
    start_date = grp["order_date"].iloc[0]
    full_index = pd.date_range(start=start_date, end=dataset_end_date, freq="D")
    n = len(full_index)
    if n <= 14:
        continue

    qty_arr = np.zeros(n, dtype=np.float32)
    wh_arr = np.full(n, np.nan, dtype=object)
    positions = (grp["order_date"] - start_date).dt.days.to_numpy(dtype=np.int32)
    qty_arr[positions] = grp["daily_quantity"].to_numpy(dtype=np.float32, copy=False)
    wh_arr[positions] = grp["warehouse_id"].to_numpy(copy=False)

    prefix = np.zeros(n + 1, dtype=np.float32)
    prefix[1:] = np.cumsum(qty_arr, dtype=np.float32)
    qty_target = prefix[15:] - prefix[1:-14]

    wh_target = np.full(n - 14, np.nan, dtype=object)
    for p in range(n - 14):
        wh_target[p] = _mode_latest(wh_arr[p + 1: p + 15])

    out_df = pd.DataFrame({
        "customer_id": customer_id,
        "item_name": item_name,
        "current_date": full_index[:n - 14],
        "quantity_next14d": qty_target,
        "will_purchase_next14d": (qty_target > 0).astype(np.int8),
        "mode_warehouse_next14d": wh_target,
    })
    target_parts.append(out_df)

target_df = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame(
    columns=["customer_id", "item_name", "current_date", "quantity_next14d",
             "will_purchase_next14d", "mode_warehouse_next14d"]
)

model_df = model_df.merge(
    target_df, on=["customer_id", "item_name", "current_date"], how="left", sort=False
)

model_df["quantity_next14d"] = model_df["quantity_next14d"].astype("float32")
model_df["will_purchase_next14d"] = model_df["will_purchase_next14d"].fillna(0).astype("int8")
model_df["mode_warehouse_next14d"] = model_df["mode_warehouse_next14d"].astype("category")

print("Target generation completed.")

# ==============================================================
# STEP 8: FEATURE AND TARGET SETUP
# ==============================================================
FEATURE_COLS = [
    "purchase_day_count_of_this_item",
    "days_since_last_order_of_this_item",
    "days_since_first_order_of_this_item",
    "last_quantity_of_this_item",
    "mean_quantity_of_this_item_till_now",
    "total_quantity_of_this_item_till_now",
    "item_quantity_last7d",
    "item_quantity_last14d",
    "item_quantity_last30d",
    "max_quantity_of_this_item",
    "avg_days_between_item_orders",
    "stop_ratio",
    "orders_since_last_purchase_of_item",
    "general_warehouse_of_this_item_till_now",
    "last_warehouse_of_this_item",
    "item_name",
]

CAT_FEATURES = [
    "item_name",
    "general_warehouse_of_this_item_till_now",
    "last_warehouse_of_this_item",
]

# ==============================================================
# STEP 9: TEMPORAL SPLIT (85 / 15)
# ==============================================================
df = model_df.copy()
df["current_date"] = pd.to_datetime(df["current_date"]).dt.normalize()

# Rows in the last 14 days of the dataset have no computable future target
# (the 14-day forward window extends beyond the data) and are reserved for
# inference only (Step 13). Filter them out before any train/val/test split
# to prevent NaN targets from leaking into y_test.
df_labeled = df[df["quantity_next14d"].notna()].reset_index(drop=True)

unique_dates = pd.Index(pd.to_datetime(df_labeled["current_date"].unique())).sort_values()
n_dates = len(unique_dates)

if n_dates < 2:
    print("[ERROR] Not enough unique dates for a temporal split.")
    sys.exit(1)

train_cut = max(1, int(n_dates * 0.85))
if train_cut >= n_dates:
    train_cut = n_dates - 1

train_end_date = unique_dates[train_cut - 1]
test_start_date = unique_dates[train_cut]

train_mask = df_labeled["current_date"] <= train_end_date
test_mask = df_labeled["current_date"] > train_end_date

# Inner validation slice from last 15% of train for early stopping
train_only_dates = unique_dates[:train_cut]
val_cut_inner = max(1, int(len(train_only_dates) * 0.85))
val_start_inner = train_only_dates[val_cut_inner]

val_mask_inner = (df_labeled["current_date"] > val_start_inner) & train_mask
train_mask_final = (df_labeled["current_date"] <= val_start_inner)

def prepare_X(subset):
    X = subset[FEATURE_COLS].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("object").where(X[c].notna(), "MISSING").astype(str)
    return X

X_train = prepare_X(df_labeled.loc[train_mask_final])
y_train = df_labeled.loc[train_mask_final, "quantity_next14d"].astype("float32")

X_val = prepare_X(df_labeled.loc[val_mask_inner])
y_val = df_labeled.loc[val_mask_inner, "quantity_next14d"].astype("float32")

X_test = prepare_X(df_labeled.loc[test_mask])
y_test = df_labeled.loc[test_mask, "quantity_next14d"].astype("float32")

print(f"\nSplit: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
print(f"  Train ends : {train_end_date.date()}")
print(f"  Test starts: {test_start_date.date()}")

# ==============================================================
# STEP 10: TRAIN QUANTITY REGRESSOR
# ==============================================================
print("\nStarting Quantity Regressor training...")

train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)
test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

quantity_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="MAE",
    iterations=300,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5.0,
    random_seed=42,
    task_type=TASK_TYPE,
    od_type="Iter",
    od_wait=250,
    verbose=0,
    allow_writing_files=False,
)

quantity_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
print("Quantity Regressor training completed.")

# ==============================================================
# STEP 11: EVALUATION ON TEST SET
# ==============================================================
test_pred = quantity_model.predict(test_pool)
test_pred_clipped = np.clip(test_pred, 0, None)

mae = mean_absolute_error(y_test, test_pred_clipped)
rmse = np.sqrt(mean_squared_error(y_test, test_pred_clipped))
r2 = r2_score(y_test, test_pred_clipped)

print("\n--- Quantity Regressor | Test Set Metrics ---")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ==============================================================
# STEP 12: WAREHOUSE-LEVEL INVENTORY EVALUATION
# ==============================================================
print("\nRunning warehouse-level inventory evaluation...")

master_eval = master_df.copy()
master_eval["order_date"] = pd.to_datetime(master_eval["order_date"]).dt.normalize()
df_eval = df_labeled.copy()

snapshot_df = df_eval.loc[
    test_mask,
    ["customer_id", "item_name", "current_date",
     "general_warehouse_of_this_item_till_now", "quantity_next14d"]
].copy()
snapshot_df["predicted_quantity"] = test_pred_clipped
snapshot_df["predicted_warehouse"] = snapshot_df["general_warehouse_of_this_item_till_now"].astype("string")
snapshot_df["item_name"] = snapshot_df["item_name"].astype("string")

pred_agg = (
    snapshot_df.groupby(
        ["current_date", "predicted_warehouse", "item_name"],
        observed=True, sort=False, as_index=False
    )["predicted_quantity"].sum()
    .rename(columns={"predicted_warehouse": "warehouse_id"})
)
pred_agg["warehouse_id"] = pred_agg["warehouse_id"].astype("string")
pred_agg["item_name"] = pred_agg["item_name"].astype("string")
pred_agg["current_date"] = pd.to_datetime(pred_agg["current_date"]).dt.normalize()

daily_actuals = (
    master_eval.groupby(
        ["order_date", "warehouse_id", "item_name"],
        observed=True, sort=False, as_index=False
    )[qty_col].sum()
    .rename(columns={"order_date": "current_date", qty_col: "daily_actual"})
)
daily_actuals["current_date"] = pd.to_datetime(daily_actuals["current_date"]).dt.normalize()
daily_actuals["warehouse_id"] = daily_actuals["warehouse_id"].astype("string")
daily_actuals["item_name"] = daily_actuals["item_name"].astype("string")

all_dates = pd.date_range(
    start=master_eval["order_date"].min(),
    end=master_eval["order_date"].max(),
    freq="D",
)
test_dates = pd.Index(pd.to_datetime(df_eval.loc[test_mask, "current_date"].unique())).sort_values()

actual_parts = []
for (warehouse_id, item_name), grp in daily_actuals.groupby(
    ["warehouse_id", "item_name"], observed=True, sort=False
):
    s = (
        grp.set_index("current_date")["daily_actual"]
        .reindex(all_dates, fill_value=0.0)
        .astype(np.float32)
    )
    arr = s.to_numpy(dtype=np.float32, copy=False)
    n = len(arr)
    prefix = np.concatenate(([0.0], np.cumsum(arr, dtype=np.float32)))
    actual_next14 = prefix[15:] - prefix[1:-14]
    pos = np.arange(n - 14, dtype=np.int32)
    prev14 = prefix[pos] - prefix[np.maximum(0, pos - 14)]
    hist_mean = np.zeros(n - 14, dtype=np.float32)
    mask_hist = pos > 0
    hist_mean[mask_hist] = (prefix[pos[mask_hist]] / pos[mask_hist]) * 14.0

    out = pd.DataFrame({
        "current_date": all_dates[:n - 14],
        "warehouse_id": warehouse_id,
        "item_name": item_name,
        "actual_quantity": actual_next14,
        "persistence_baseline": prev14,
        "historical_mean_baseline": hist_mean,
    })
    out = out[out["current_date"].isin(test_dates)].copy()
    actual_parts.append(out)

actual_agg = (
    pd.concat(actual_parts, ignore_index=True) if actual_parts
    else pd.DataFrame(columns=["current_date", "warehouse_id", "item_name",
                                "actual_quantity", "persistence_baseline", "historical_mean_baseline"])
)
actual_agg["warehouse_id"] = actual_agg["warehouse_id"].astype("string")
actual_agg["item_name"] = actual_agg["item_name"].astype("string")
actual_agg["current_date"] = pd.to_datetime(actual_agg["current_date"]).dt.normalize()

inventory_eval_df = actual_agg.merge(
    pred_agg, on=["current_date", "warehouse_id", "item_name"], how="outer"
)
for col in ["actual_quantity", "predicted_quantity", "persistence_baseline", "historical_mean_baseline"]:
    if col in inventory_eval_df.columns:
        inventory_eval_df[col] = inventory_eval_df[col].fillna(0.0).astype(np.float32)

micro_actual = inventory_eval_df["actual_quantity"].sum()
micro_pred = inventory_eval_df["predicted_quantity"].sum()
micro_vfr = (
    np.minimum(inventory_eval_df["predicted_quantity"], inventory_eval_df["actual_quantity"]).sum() / micro_actual
    if micro_actual > 0 else np.nan
)
micro_osr = (
    np.maximum(inventory_eval_df["predicted_quantity"] - inventory_eval_df["actual_quantity"], 0.0).sum() / micro_actual
    if micro_actual > 0 else np.nan
)
micro_bias = (micro_pred - micro_actual) / micro_actual if micro_actual > 0 else np.nan
micro_model_mae = np.abs(inventory_eval_df["predicted_quantity"] - inventory_eval_df["actual_quantity"]).sum()
micro_persistence_mae = np.abs(inventory_eval_df["persistence_baseline"] - inventory_eval_df["actual_quantity"]).sum()
micro_hist_mean_mae = np.abs(inventory_eval_df["historical_mean_baseline"] - inventory_eval_df["actual_quantity"]).sum()
micro_mase_persistence = np.nan if micro_persistence_mae == 0 else micro_model_mae / micro_persistence_mae
micro_mase_hist_mean = np.nan if micro_hist_mean_mae == 0 else micro_model_mae / micro_hist_mean_mae

print("\n--- Warehouse-Item Inventory | Test Set Metrics ---")
print(f"  Micro VFR (higher better)              : {micro_vfr:.4f}")
print(f"  Micro OSR (lower better)               : {micro_osr:.4f}")
print(f"  Forecast Bias (closer to 0 better)     : {micro_bias:.4f}")
print(f"  MASE vs Persistence (lower better)     : {micro_mase_persistence:.4f}")
print(f"  MASE vs Historical Mean (lower better) : {micro_mase_hist_mean:.4f}")
print(f"  Total actual volume                    : {micro_actual:.2f}")
print(f"  Total predicted volume                 : {micro_pred:.2f}")

print("\nEvaluation completed.")

# ==============================================================
# STEP 13: BUILD FINAL INFERENCE OUTPUT
# Treat TRUE_MAX_DATE as today.
# Use the most recent feature state per customer-item
# (rows from the last 14 days of the full dataset).
# ==============================================================
print("\nBuilding final 14-day demand predictions for all customer-item pairs...")

# Take the most recent feature state per customer-item from the full panel
# (i.e. the row at TRUE_MAX_DATE, or the closest prior date with features)
last_rows = (
    df.sort_values("current_date")
    .groupby(["customer_id", "item_name"], observed=True)
    .last()
    .reset_index()
)

X_infer = prepare_X(last_rows)
infer_pool = Pool(X_infer, cat_features=CAT_FEATURES)
infer_pred = np.clip(quantity_model.predict(infer_pool), 0, None).astype(np.float32)

# ==============================================================
# OUTPUT 1: Customer-Item level predictions
# ==============================================================
customer_item_output = pd.DataFrame({
    "customer_id": last_rows["customer_id"].values,
    "item_name": last_rows["item_name"].astype(str).values,
    "warehouse_id": last_rows["general_warehouse_of_this_item_till_now"].astype(str).values,
    f"predicted_qty_{(TRUE_MAX_DATE + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}_to_{(TRUE_MAX_DATE + pd.Timedelta(days=14)).strftime('%Y-%m-%d')}": infer_pred,
})

# ==============================================================
# OUTPUT 2: Warehouse-Item aggregated inventory plan
# Aggregate from the FULL customer-item data before any filtering,
# so that small per-customer quantities (e.g. 30 customers x 0.3 units)
# are correctly summed into the warehouse total before the <1 check.
# ==============================================================
pred_col = customer_item_output.columns[-1]

warehouse_item_output = (
    customer_item_output.groupby(["warehouse_id", "item_name"], sort=False)[pred_col]
    .sum()
    .reset_index()
    .sort_values(pred_col, ascending=False)
    .reset_index(drop=True)
)

# --- Clean up both outputs ---
# Round first, then filter, so CSV values and the threshold are consistent.
customer_item_output[pred_col] = customer_item_output[pred_col].round(2)
warehouse_item_output[pred_col] = warehouse_item_output[pred_col].round(2)

# Drop rows whose rounded predicted demand is below 1 unit.
customer_item_output = (
    customer_item_output[customer_item_output[pred_col] >= 1]
    .reset_index(drop=True)
)
warehouse_item_output = (
    warehouse_item_output[warehouse_item_output[pred_col] >= 1]
    .reset_index(drop=True)
)

print(f"\nCustomer-item predictions built: {len(customer_item_output):,} rows")
print(f"Warehouse-item aggregation built: {len(warehouse_item_output):,} rows")



# ==============================================================
# STEP 14: SAVE OUTPUTS
# ==============================================================
save_choice = input("\nSave output CSVs? (1 = Yes, 0 = No): ").strip()

if save_choice == "1":
    cust_path = input(
        "Enter path for customer-item CSV [press Enter for 'customer_item_prediction_14_days.csv']: "
    ).strip()
    cust_path = cust_path if cust_path else "customer_item_prediction_14_days.csv"

    wh_path = input(
        "Enter path for warehouse-item CSV [press Enter for 'inventory_14_days_FINAL.csv']: "
    ).strip()
    wh_path = wh_path if wh_path else "inventory_14_days.csv"

    customer_item_output.to_csv(cust_path, index=False)
    warehouse_item_output.to_csv(wh_path, index=False)

    print(f"\nCustomer-item predictions saved to : {cust_path}")
    print(f"Warehouse-item inventory saved to  : {wh_path}")
else:
    print("Outputs not saved.")

print("\nDone.")
