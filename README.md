> **Note:** The original implementation of this project has been preserved under `version1_archive/`. The current repository root contains a complete redesign with a new modeling pipeline and improved forecasting methodology.

# Predictive Framework for B2B Grocery Supply Chain Optimization
### UNORG — Demand Forecasting & Inventory Planning System

---

## Background

UNORG is a B2B grocery distribution platform serving a diverse customer base across multiple warehouses. The business operates in a high-SKU, high-frequency environment where purchasing patterns vary widely across customers, items, and time. Inventory decisions at UNORG are consequential: understocking leads to lost sales and dissatisfied customers, while overstocking ties up working capital and strains warehouse operations.

The core operational challenge was twofold. First, the planning team had no reliable way to anticipate *when* a customer would place their next order, making proactive outreach and delivery scheduling difficult. Second, warehouse managers lacked a data-driven view of *how much* of each SKU to stock over the coming two weeks, forcing conservative over-ordering as a default buffer.

This project addresses both problems through a bottom-up predictive framework: customer behavior is modeled first, and warehouse-level inventory requirements are derived from those customer-level predictions.

---

## Problem Statement

1. **Customer Order Prediction**: Given a customer and the current date, predict the probability that the customer places an order on each of the next 14 days.
2. **SKU-Level Demand Forecasting**: Given a warehouse-item pair, forecast the quantity required over the next 14 days to meet expected customer demand.
3. **Inventory Planning**: Aggregate customer-item forecasts into a actionable two-week stocking plan per warehouse, balancing stockout risk against overstock cost.

---

## Dataset

Two raw datasets covering the last six months of transactional activity were used:

### 1. Order Data (Last Six Months)
- **Rows**: 64,459
- **Columns**: 14
- **Key Fields**:
  - `order_date`, `order_id`, `order_number`
  - `customer_id`, `customer_name`, `poc_name`
  - `amount`, `discount`, `net_order_amount`, `profit`
  - `order_status`, `warehouse_name`

### 2. Order Item Data (Last Six Months)
- **Rows**: 115,093
- **Columns**: 11
- **Key Fields**:
  - `order_id`, `order_item_id`, `item_name`
  - `quantity`, `invoiced_quantity`
  - `mrp`, `price_per_unit`, `amount`, `discount_amount`, `profit`

The two tables are linked via `order_id`. The item dataset covers approximately 600 unique SKUs across multiple warehouses.

---

## Solution Methodology

### Part 1 — Customer Order Prediction

The customer prediction task was framed as a **customer-day panel problem**. Every customer was expanded across every date in the observation window, creating one training instance per customer-date. This framing allowed the model to answer: *"Given everything known up to today, what will this customer do over the next 14 days?"*

Careful leakage control was applied throughout feature construction to ensure no future information was used as input.

#### Feature Engineering

Features were designed to capture four behavioral dimensions:

- **Recency**: Days since last 1st, 2nd, and 3rd prior orders
- **Frequency**: Rolling order counts over 7-day, 14-day, and 30-day windows, plus ratios between windows
- **Monetary**: Net order amount from previous 1, 2, and 3 orders; mean historical net order amount
- **Temporal/Behavioral**: Day of week, month, weekend flag, peak order weekday, days until next peak weekday, customer tenure, total historical orders

SHAP analysis confirmed that the strongest predictors were `frequency_30d`, `total_orders`, `frequency_14d`, `days_since_first_order`, `days_since_last_order`, `mean_net_order_amount`, and `frequency_7d`. Temporal and rhythm features contributed marginally in comparison.

#### Modeling — Stage 1: Direct Multi-Label Profile Prediction

The first attempt modeled the 14-day order profile directly as a multi-label classification problem using a CatBoost classifier, predicting each day's binary order indicator simultaneously.

| Model | Tolerant Jaccard | Notes |
|---|---|---|
| Direct Multi-Label CatBoost Classifier | 0.56 | Baseline; predicts all 14 days jointly |

While usable, this approach lacked any mechanism to coordinate the total number of predicted order days with actual customer order frequency.

#### Modeling — Stage 2: Count Prediction + Profile Distribution

The pipeline was redesigned into two stages. First, a CatBoost regressor predicts the *total number of orders* a customer will place over the next 14 days. This count is then used to distribute predicted orders across the 14-day window. Six distribution architectures were designed and benchmarked:

**Stage 2a — Count Regressor**

| Model | MAE | RMSE | Within-±1 Accuracy |
|---|---|---|---|
| CatBoost Order Count Regressor | 0.81 | 1.15 | **86.6%** |

**Stage 2b — Profile Distribution Architectures**

| Architecture | Tolerant Jaccard | Notes |
|---|---|---|
| Sequential Day Classifier Chain | 0.4717 | Predicts each day conditioned on prior day predictions |
| Budget-Aware Sequential Chain | **0.5935** | Best overall; explicitly tracks remaining order budget per step |
| Pairwise Day-Ranking Model (CatBoost Ranker) | 0.3796 | Ranks 14 candidate days; best ranking metrics but weaker profile overlap |
| Independent Day Scorer with Top-K Selection (MLP) | 0.3630 | Scores all 14 days independently; selects top-k using count prediction |
| Competitive Day Scorer with Top-K Selection (Softmax MLP) | 0.3614 | Forces competition among days via softmax; top-k selection at inference |
| Competitive Day Scorer with Threshold Tuning (Softmax MLP) | 0.5835 | Same softmax architecture; threshold tuned per predicted count |

The **Budget-Aware Sequential Chain** was selected as the champion model, achieving Tolerant Jaccard of **0.594**. Notably, the Threshold-Tuned Softmax MLP independently converged to 0.584 — a very similar ceiling despite being a fundamentally different architecture. Both represent only a modest improvement over the direct multi-label baseline of 0.56. This convergence across architectures strongly suggests an **information ceiling**: the available feature set, not the model design, is the primary bottleneck on day-level timing prediction.

---

### Part 2 — Inventory Forecasting

The inventory problem was approached from two directions and the results compared rigorously.

#### Approach A: Bottom-Up Customer Aggregation

Customer-level quantity predictions were generated using a CatBoost regressor trained on the engineered customer-day panel (MAE 2.30, RMSE 10.64 on test). These predictions were then aggregated to the warehouse-item level to produce inventory requirements.

This approach captured **91.4% of predictable demand** — that is, 91.4% of the future demand that was representable given the customer-item pairs visible at prediction time. The remaining ~27.5% of total future demand came from previously unseen customer-item pairs that were structurally invisible to the customer model.

| Metric | Value |
|---|---|
| Total Actual Volume | 1,491,305 units |
| Total Predicted Volume | 1,575,031 units |
| Useful Quantity (fill) | 988,024 units |
| Excess Quantity (overstock) | 587,008 units |
| Missed Quantity (stockout) | 503,281 units |
| Micro Volume Fill Rate | 0.672 |
| Micro Overstock Ratio | 0.413 |
| Forecast Bias | 0.085 |

#### Approach B: Direct Warehouse-Item Regression

A CatBoost regressor was also trained directly on the warehouse-item time series, using rolling demand windows, lag features, recency signals, global item demand, and local unique-customer activation counts.

| Metric | Value |
|---|---|
| Micro Volume Fill Rate | 0.655 |
| Micro Overstock Ratio | 0.497 |
| Forecast Bias | 0.152 |
| MASE vs. Persistence | 1.192 |
| MASE vs. Historical Mean | 1.014 |

The customer aggregation approach outperformed direct warehouse regression on every business-relevant metric — even though ~27.5% of future demand came from new customer-item pairs that were structurally invisible to the customer model at prediction time. The reason it still won: within the 72.5% of demand it could represent, it captured 91.4% of that demand accurately. By preserving customer-level heterogeneity, aggregation also benefits from partial error cancellation across hundreds of independent predictions — an advantage the compressed warehouse time series cannot replicate. The warehouse model, by contrast, performed adequately on steady-demand SKUs but consistently underforecast demand explosions and overforecast zero-demand rows, with the latter alone driving 40.4% of total excess inventory.

#### SKU-Level Findings

The hardest SKUs to forecast were high-volume oils and staples — particularly Ruchi Gold Palm Pouch (1L), Oil Ruchi Gold Palm Pouch (770Gm), and several soya and flour items. Demand spikes on these SKUs were not driven by a single large customer but by many customers becoming simultaneously active. No historical demand signal reliably predicted these activation events, because the underlying drivers (promotions, pricing, seasonal stocking) were absent from the dataset.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| **Micro Volume Fill Rate (VFR)** | Fraction of actual demand covered by predictions, averaged across warehouse-item pairs |
| **Micro Overstock Ratio (OSR)** | Fraction of predicted volume that exceeds actual demand |
| **Tolerant Jaccard** | Day-level overlap between predicted and actual order profile, with ±1 day tolerance |
| **Strict Jaccard** | Exact day-level overlap with no tolerance |
| **MASE** | Mean Absolute Scaled Error relative to a naive baseline (persistence or historical mean) |
| **Forecast Bias** | Signed relative difference between total predicted and actual volume |
| **MAE / RMSE** | Standard regression error on order quantity |
| **ROC-AUC / PR-AUC** | Ranking quality of the purchase probability classifier |

---

## Key Findings

1. **Order count is far easier to predict than exact timing.** The CatBoost count regressor achieved 86.6% within-±1 accuracy, while no timing model exceeded Tolerant Jaccard of 0.594 — reflecting genuine day-level uncertainty in customer behavior.

2. **Global coordination among days matters more than independent thresholds.** The budget-based sequential chain outperformed independent per-day classifiers precisely because it explicitly constrained total predicted orders to match the count forecast.

3. **Bottom-up aggregation outperforms direct warehouse regression on this dataset.** Customer-level predictions preserve behavioral heterogeneity and benefit from partial error cancellation when aggregated — advantages that the compressed warehouse time series cannot replicate.

4. **27.5% of demand is structurally invisible to the customer model.** Demand from new customer-item pairs cannot be recovered from purchase history alone and requires complementary signals such as assortment data or new customer onboarding signals.

5. **Forecasting failures are concentrated, not random.** The worst errors occurred on a small set of explosive SKUs driven by synchronized customer activation events — a pattern that historical demand history alone cannot predict without promotional, pricing, or event data.

---

## Limitations

- **No external signals**: The dataset contains no information on promotions, pricing changes, holidays, supplier lead times, stockouts, or marketing events. These are the primary drivers of demand spikes and their absence is the main ceiling on forecast accuracy.
- **New customer-item pairs**: Approximately 27.5% of future demand comes from customer-item combinations not present in historical data, which is structurally unforecastable under the current formulation.
- **Sparse warehouse histories**: 123 of 953 warehouse-item pairs had fewer than 60 rows of history, limiting the reliability of rolling and lag features for those series.
- **Six-month observation window**: A longer history would improve seasonal signal detection and reduce sensitivity to short-term anomalies.
- **No product-level quantity model at customer resolution**: The current pipeline forecasts total order quantity per customer, not which specific SKUs they will order. Item-level customer prediction is the natural next extension.

---

## Repository Structure

```
.
├── order_data_last_six_month.csv.zip           # Raw order-level data (64,459 rows)
├── associated_order_item_data_last_six_month.csv.zip  # Item-level order breakdown (115,093 rows)
├── customer_order_prediction.ipynb             # Customer purchase profile modeling
├── inventory_planning.ipynb                    # SKU-level demand forecasting and inventory planning
├── customer_14day_profile.csv.zip                 # Customer-wise 14-day order profile
├── customer_item_prediction_14_days.csv        # Forecasted daily SKU quantities per customer (14 days)
├── inventory_14_days.csv                 # Aggregated warehouse inventory plan (14 days)
├── order_profile.py                              # Standalone script: outputs day-wise purchase profile per customer
└── inventory.py                               # Standalone script: outputs per-customer SKU forecast + inventory plan
```

---

## Usage

### Customer Order Prediction
Run `order_profile.py` to generate a binary 14-day purchase profile per customer, with one row per customer and 14 day columns (0/1).

```bash
python order_profile.py
```

### Inventory Planning
Run `inventory.py` to generate both the customer-item quantity forecast and the aggregated warehouse stocking plan.

```bash
python inventory.py
```

This outputs two files:
- Per-customer SKU quantity forecast for the next 14 days
- Aggregated inventory requirement plan per warehouse

---

## Business Impact

| Use Case | Model | Action |
|---|---|---|
| Proactive customer outreach | Customer order model | Target high-probability buyers before order window |
| Delivery route optimization | Customer order model | Schedule routes around predicted active customers |
| Demand-driven procurement | Inventory model | Trigger replenishment orders ahead of demand |
| Stockout prevention | Inventory model | Flag high-risk SKU-warehouse pairs for priority stocking |
| Overstock reduction | Inventory model | Suppress unnecessary buffer stock on low-demand pairs |

---

## Tech Stack

- **Modeling**: CatBoost, scikit-learn, PyTorch
- **Feature Engineering**: pandas, NumPy
- **Interpretability**: SHAP
- **Forecasting Baselines**: Prophet, Holt-Winters
- **Evaluation**: Custom metrics (VFR, OSR, Tolerant Jaccard, MASE)

---

