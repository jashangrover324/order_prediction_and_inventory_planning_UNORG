import os

def main():
    import os
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    # Step 1: Ask user to paste the file path
    file_path = input("Paste the full path of the customer order file (.csv): ").strip()

    # Optional: Basic check if the file exists
    if not os.path.isfile(file_path):
        print("File not found. Please check the path and try again.")
        return

    # Step 2: Load the data (example: assuming it's a CSV)
    try:
        orders = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Step 1: Ask user to paste the file path
    file_path1 = input("Paste the full path of the item order file (.csv): ").strip()

    # Optional: Basic check if the file exists
    if not os.path.isfile(file_path):
        print("File not found. Please check the path and try again.")
        return

    # Step 2: Load the data (example: assuming it's a CSV)
    try:
        order_items = pd.read_csv(file_path1)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import timedelta
        

    print("Started working")

    #SKU FORECAST AND INVENTORY
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import ttest_rel
    from datetime import timedelta
    import warnings
    warnings.filterwarnings("ignore")

    # Load data
    orders = pd.read_csv("order_data_last_six_month.csv")
    order_items = pd.read_csv("associated_order_item_data_last_six_month.csv")

    # Parse dates
    orders["order_date"] = pd.to_datetime(orders["order_date"], dayfirst=True)

    # Merge orders and items
    data = pd.merge(order_items, orders[['order_date','customer_id','order_id','warehouse_name','warehouse_id']], on="order_id",how='left')
    data.drop(columns=['order_number','order_item_id'], inplace=True)

    import pandas as pd
    import numpy as np
    from catboost import CatBoostRegressor, Pool
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # # Step 1: Filter to last 120 days (retain original data)
    cutoff_date = data['order_date'].max() - pd.Timedelta(days=150)
    data = data[data['order_date'] >= cutoff_date]

    # Step 2: Aggregate by day per customer-item-warehouse
    daily = data.groupby(['customer_id', 'item_name', 'warehouse_id', 'order_date']).agg({
        'quantity': 'sum',
        'price_per_unit': 'mean',
        'discount_amount': 'sum',
        'amount': 'sum',
        'profit': 'sum'
    }).reset_index()

    # Step 3: Filter only active pairs
    active_pairs = daily.groupby(['customer_id', 'item_name', 'warehouse_id']).filter(lambda x: len(x) >= 10)

    # Step 4: Feature engineering (add more features)
    def create_features(group):
        group = group.set_index('order_date').asfreq('D').fillna(0)
        group['rolling_7'] = group['quantity'].rolling(7).mean().fillna(0)
        group['rolling_14'] = group['quantity'].rolling(14).mean().fillna(0)
        group['rolling_28'] = group['quantity'].rolling(28).mean().fillna(0)

        group['rolling_price'] = group['price_per_unit'].rolling(7).mean().fillna(0)
        group['rolling_discount'] = group['discount_amount'].rolling(7).sum().fillna(0)

        # Time-related features
        group['dayofweek'] = group.index.dayofweek
        group['month'] = group.index.month
        group['is_weekend'] = (group['dayofweek'] >= 5).astype(int)

        # Lag features
        group['lag_quantity_1'] = group['quantity'].shift(1).fillna(0)
        group['lag_quantity_7'] = group['quantity'].shift(7).fillna(0)
        group['lag_quantity_14'] = group['quantity'].shift(14).fillna(0)
        group['lag_quantity_30'] = group['quantity'].shift(30).fillna(0)

        # More advanced features
        group['trend'] = np.arange(len(group))
        group['rolling_var'] = group['quantity'].rolling(7).var().fillna(0)
        group['rolling_skew'] = group['quantity'].rolling(7).skew().fillna(0)

        group = group.reset_index()
        return group

    frames = []
    for (cid, item, wid), group in active_pairs.groupby(['customer_id', 'item_name', 'warehouse_id']):
        feats = create_features(group)
        feats['customer_id'] = cid
        feats['item_name'] = item
        feats['warehouse_id'] = wid
        frames.append(feats)

    features_df = pd.concat(frames, ignore_index=True)
    features_df = features_df.dropna()

    # Step 5: Log-transform target variable
    features_df['log_quantity'] = np.log1p(features_df['quantity'])

    # Step 6: Train/test split
    X = features_df[['rolling_7', 'rolling_14', 'rolling_28', 'rolling_price', 'rolling_discount', 'dayofweek',
                     'month', 'is_weekend', 'lag_quantity_1', 'lag_quantity_7', 'lag_quantity_14', 'lag_quantity_30',
                     'trend', 'rolling_var', 'rolling_skew']]
    y = features_df['log_quantity']
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.2)

    import os
    import torch
    cuda_available = os.system("nvidia-smi") == 0

    # Set task_type and devices based on the availability of CUDA/MPS or default to CPU
    if cuda_available:
        device = 'GPU'
        print("This code will run on GPU")
        devices = '0'  # Assuming GPU 0 for CUDA
    else:
        device = 'CPU'
        print("Due to unavailability of GPU, this code will run on CPU and may take longer")
        devices = ''  # No device necessary for CPU
        
    # Step 7: CatBoost Model on GPU (with hyperparameter tuning)
    catboost_model = CatBoostRegressor(iterations=1500,
                                       learning_rate=0.08,
                                       depth=8,
                                       loss_function='RMSE',
                                       task_type=device,  # Enable GPU
                                       cat_features=[],  # No categorical features for now
                                       random_seed=42,
                                       verbose=False)

    catboost_model.fit(X_train, y_train)


    # Step 9: Predict next 14-day quantity
    latest_feats = features_df.groupby(['customer_id', 'item_name', 'warehouse_id']).tail(1).copy()
    X_pred = latest_feats[['rolling_7', 'rolling_14', 'rolling_28', 'rolling_price', 'rolling_discount', 'dayofweek',
                           'month', 'is_weekend', 'lag_quantity_1', 'lag_quantity_7', 'lag_quantity_14', 'lag_quantity_30',
                           'trend', 'rolling_var', 'rolling_skew']]
    latest_feats['predicted_log_quantity'] = catboost_model.predict(X_pred)
    latest_feats['predicted_quantity'] = np.expm1(latest_feats['predicted_log_quantity']) * 14  # Forecast for next 14 days
    latest_feats['predicted_quantity'] = latest_feats['predicted_quantity'].clip(lower=0).round(2)

    # Final output
    prediction_df = latest_feats[['customer_id', 'item_name', 'warehouse_id', 'predicted_quantity']]

    # Aggregate predicted quantities by item and warehouse
    inventory_df = prediction_df.groupby(['item_name', 'warehouse_id'])['predicted_quantity'].sum().reset_index()

    # Optional: sort it for better readability
    inventory_df = inventory_df.sort_values(['warehouse_id', 'item_name']).reset_index(drop=True)

    input_dir = os.path.dirname(file_path)
    output_path1 = os.path.join(input_dir, "customer_item_prediction_14_days.csv")
    output_path2 = os.path.join(input_dir, "inventory_14_days.csv")
    try:
        prediction_df.to_csv(output_path1, index=False)
        print(f"\nAnalysis complete. Output saved to:\n{output_path1}")
    except Exception as e:
        print(f"Error saving output: {e}")

    try:
        inventory_df.to_csv(output_path2, index=False)
        print(f"\nAnalysis complete. Output saved to:\n{output_path2}")
    except Exception as e:
        print(f"Error saving output: {e}")


if __name__ == "__main__":
    main()
