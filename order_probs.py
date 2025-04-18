import os

def main():
    import os
    import pandas as pd
    # Step 1: Ask user to paste the file path
    file_path = input("Paste the full path of the file to be analyzed: ").strip()

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
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import timedelta
    print("Started working")
    


    orders['order_date'] = pd.to_datetime(orders['order_date'], dayfirst=True)

    daily_orders = (
        orders.groupby(['customer_id', 'order_date'], as_index=False)
        .agg(
            num_orders=('order_id', 'count'),  # or just use 1 if only one row per day is needed
            net_order_amount=('net_order_amount', 'sum'),
            net_profit= ('profit', 'sum')
        )
        .rename(columns={'order_date': 'date'})
    )

    #unique customers and date range
    unique_customers = daily_orders['customer_id'].unique()
    min_date = daily_orders['date'].min()
    max_date = daily_orders['date'].max()
    date_range = pd.date_range(start=min_date, end=max_date+pd.Timedelta(days=1), freq='D')

    #cartesian matrix (every customer every date)
    customer_df = pd.MultiIndex.from_product(
        [unique_customers, date_range], names=['customer_id', 'date']
    ).to_frame(index=False)

    customer_df = customer_df.merge(daily_orders, on=['customer_id', 'date'], how='left')
    customer_df= customer_df.fillna(0)

    df_daily=customer_df.copy()
    df_daily['order_placed'] = (df_daily['net_order_amount'] > 0).astype(int)

    # RFM Features

    df_daily['amount_30d'] = df_daily.groupby('customer_id')['net_order_amount'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    df_daily['amount_90d'] = df_daily.groupby('customer_id')['net_order_amount'].transform(lambda x: x.rolling(window=90, min_periods=1).sum())

    df_daily['frequency_7d'] = df_daily.groupby('customer_id')['order_placed'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df_daily['frequency_30d'] = df_daily.groupby('customer_id')['order_placed'].transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    df_daily['frequency_90d'] = df_daily.groupby('customer_id')['order_placed'].transform(lambda x: x.rolling(window=90, min_periods=1).sum())

    # Recency is tricky without an actual last order date, so we'll use a placeholder
    # and will need to calculate it after more sophisticated data preparation.
    df_daily['recency'] = 0  # Placeholder


    # Time-Based Features
    df_daily['month'] = df_daily['date'].dt.month


    # Rolling Statistics

    df_daily['amount_rolling_mean_30d'] = df_daily.groupby('customer_id')['net_order_amount'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df_daily['amount_rolling_mean_90d'] = df_daily.groupby('customer_id')['net_order_amount'].transform(lambda x: x.rolling(window=90, min_periods=1).mean())

    df_daily['amount_rolling_std_90d'] = df_daily.groupby('customer_id')['net_order_amount'].transform(lambda x: x.rolling(window=90, min_periods=1).std())


    # Create an empty column to store last order date
    df_daily['last_order_date'] = df_daily.groupby('customer_id')\
        .apply(lambda g: g['date'].where(g['order_placed'] == 1).ffill())\
        .reset_index(level=0, drop=True)

    # Now calculate recency as the difference in days between current date and last order date
    df_daily['recency'] = (df_daily['date'] - df_daily['last_order_date']).dt.days

    cols_to_shift = [
        'amount_30d', 'amount_90d',
        'frequency_7d', 'frequency_30d', 'frequency_90d','recency',
        'amount_rolling_mean_30d','amount_rolling_mean_90d',
        'amount_rolling_std_90d'
    ]

    # Shift each column by 1 row within each customer_id group
    df_daily[cols_to_shift] = df_daily.groupby('customer_id')[cols_to_shift].shift(1)
    df_daily.fillna(0)

    df_daily['recency']=df_daily['recency'].fillna(999)
    df_daily=df_daily.fillna(0)

    pred_df = df_daily.groupby('customer_id', group_keys=False).tail(1)

    # Step 2: Remove these rows from df_daily
    df_daily = df_daily.drop(pred_df.index)

    from sklearn.model_selection import train_test_split


    # Define features and target
    X = df_daily.drop(columns=['customer_id', 'date','num_orders',	'net_order_amount',	'net_profit', 'order_placed','last_order_date'])
    pred_df=pred_df.drop(columns=['date','num_orders',	'net_order_amount',	'net_profit', 'order_placed','last_order_date'])
    y = df_daily['order_placed']


    # Train/eval split
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Get unique class labels
    classes = np.unique(y_train)

    # Compute weights
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Convert to list if needed
    weights = weights.tolist()

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

    print("Model Training started")
    from catboost import CatBoostClassifier
    from sklearn.metrics import classification_report, roc_auc_score

    # Optional: If you have categorical columns
    categorical_features = ['day_of_week','month']  # Example: ['customer_id', 'day_of_week']

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.001,
        depth=10,
        loss_function='Logloss',
        task_type=device,
        devices='0',
        class_weights=weights,
        verbose=100,
        random_seed=42
    )


    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=(X_eval, y_eval),
        early_stopping_rounds=50
    )

    pred_df['Probability of order']=model.predict_proba(pred_df.drop(columns=['customer_id']))[:, 1]
    prob_order= pred_df[['customer_id','Probability of order']]
    
    input_dir = os.path.dirname(file_path)
    output_path = os.path.join(input_dir, "Order_probabilities.csv")

    try:
        prob_order.to_csv(output_path, index=False)
        print(f"\nAnalysis complete. Output saved to:\n{output_path}")
    except Exception as e:
        print(f"Error saving output: {e}")

if __name__ == "__main__":
    main()
