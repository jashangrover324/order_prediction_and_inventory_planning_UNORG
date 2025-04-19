
# 📦 Predictive Framework for Grocery Supply Chain Optimization

## 🚀 Project Overview

UNORG, a B2B grocery platform, serves a diverse customer base with dynamic purchasing patterns across a vast SKU inventory. This project presents an integrated predictive framework aimed at improving:

- **Customer-level order predictability**
- **SKU-level demand forecasting**
- **Inventory Planning**

The goal: reduce **stockouts**, minimize **overstocking**, and enhance **customer satisfaction** through data-driven supply chain management.

---

## 📂 Project Files Description

- **`order_data_last_six_month.csv.zip`**: Compressed file containing raw order-level data for the past six months.
- **`associated_order_item_data_last_six_month.csv.zip`**: Contains item-wise order breakdown for each order in the same period.
- **`customer_order_prediction.ipynb`**: Jupyter notebook implementing the customer daily order prediction model.
- **`inventory_planning.ipynb`**: Jupyter notebook for SKU-level demand forecasting and inventory planning.
- **`Order_probabilities.csv.zip`**: Final prediction results with customer-wise order probabilities.
- **`customer_item_prediction_14_days.csv`**: Forecasted daily SKU quantities per customer for the upcoming 14 days.
- **`inventory_14_days_FINAL.csv`**: Aggregated inventory requirement plan for the next two weeks.
- **`order_probs.py`**: Script wrapping the customer order prediction model for local or automated execution. Outputs a .csv file with order probabilities for each customer.
- **`inventory.py`**: Script wrapping the item order prediction model and inventory planning for local or automated execution. Outputs two .csv files, with forecasted daily SKU quantities per customer and aggregated inventory requirement plan for the upcoming 14 days.


## 📊 Dataset Overview

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

---

## 🎯 Problem Statement

Design a predictive framework to:
1. Forecast **daily customer order probabilities**
2. Estimate **SKU-level demand for the next day**
3. Generate a **two-week inventory stocking plan**

---

## 🤖 Predictive Modeling

### 1. Customer Order Prediction Model
**Goal**: Predict if a customer will place an order on a given day.

#### Feature Engineering:
- **RFM Features**: Recency, Frequency (7/30/90d), Monetary values
- **Temporal**: Month, Day of Week, Weekend Flag
- **Rolling & Lag Features**: 7-day mean/std, lag 1–7d for amounts

#### Algorithms:
- **Baseline**: Logistic Regression
- **Advanced**: Random Forest, CatBoost

#### Output:
- Binary classification (Order/No Order)
- Probability score for flexible targeting

#### Evaluation Metrics:
- **CatBoost Classifier**
  | Dataset     | Accuracy | ROC-AUC |
  |-------------|----------|---------|
  | Train Set   | 0.7786   | 0.8041  |
  | Test Set    | 0.7802   | 0.8045  |

---

### 2. Inventory Forecasting Model
**Goal**: Predict quantity of each SKU to be ordered per store for the next day.

#### Feature Engineering:
- **Temporal**: Day, Month, Weekend, Holidays
- **Demand History**: Rolling stats over 7/14/30 days
- **Lag Features**: Lag 1/7/14 days
- **Stock Signals**: Lead time, stockouts, average quantity/order

#### Algorithms:
- **LightGBM**, **XGBoost**

#### Evaluation Metrics:
- **LightGBM**
  | Dataset     | MAE   | RMSE   | R²     |
  |-------------|-------|--------|--------|
  | Train Set   | 2.10  | 77.08  | 0.6069 |
  | Test Set    | 1.48  | 23.84  | 0.8813 |

- **XGBoost**
  | Dataset     | MAE   | RMSE   | R²     |
  |-------------|-------|--------|--------|
  | Train Set   | 1.50  | 34.64  | 0.9206 |
  | Test Set    | 2.00  | 27.04  | 0.85   |

---

## 📈 Key Insights

### 🔍 Feature Importance
- **Customer Model**: `frequency_30d`, `recency`, `month` are top signals.
- **Inventory Model**: `avg_quantity_per_order`, `order_count`, and recent rolling averages dominate.

### 🏆 Performance Summary
- High alignment between predicted and actual values.
- Low error margins (±0.5 units) even for niche SKUs.
- Scalable across bulk and retail segments.

---

## 🛒 SKU & Customer Demand Forecasting

### Top Forecasted SKUs:
1. **Ruchi Gold Palm Pouch (1L)** – 1.35M units
2. **Normal Sugar**
3. **Tata Salt (1kg)**  
> Oils dominate the top 10, making up 70% of high-volume SKUs.

### Customer Forecasting:
- Most daily demands fall in **1–30 units** range.
- Some SKUs exceed **100 units/day**, hinting at wholesale behavior.
- Forecasts allow personalized targeting and hyper-local stocking.

---

## 🧠 Deployment & Use Cases

| Use Case | Model | Action |
|----------|-------|--------|
| Targeted Marketing | Customer Model | Engage high-probability buyers |
| Delivery Optimization | Customer Model | Plan efficient routes |
| Demand-Driven Procurement | Inventory Model | Trigger replenishment orders |
| Stockout & Overstock Control | Inventory Model | Maintain optimal safety stock |

---

## 📌 Summary

By combining customer behavior modeling with SKU-level forecasting, **C-Sharp** enables precision in grocery logistics, contributing to:
- Higher **conversion rates**
- Lower **logistics costs**
- **Smoother warehouse operations**
- **Smarter inventory planning**
