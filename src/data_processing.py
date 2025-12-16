import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load raw data
RAW_DATA_PATH = '../data/raw/training.csv'  # Adjust if needed
df = pd.read_csv(RAW_DATA_PATH)
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# 1. Aggregate Features per CustomerId
agg_features = df.groupby('CustomerId').agg(
    total_amount=('Amount', 'sum'),
    avg_amount=('Amount', 'mean'),
    transaction_count=('TransactionId', 'count'),
    std_amount=('Amount', 'std'),
    total_value=('Value', 'sum'),
    avg_value=('Value', 'mean'),
).reset_index()

agg_features['std_amount'] = agg_features['std_amount'].fillna(0)

# 2. Time-based features (from TransactionStartTime)
time_features = df.groupby('CustomerId').agg(
    transaction_hour_mean=('TransactionStartTime', lambda x: x.dt.hour.mean()),
    transaction_day_mean=('TransactionStartTime', lambda x: x.dt.day.mean()),
    transaction_month=('TransactionStartTime', lambda x: x.dt.month.mode()[0] if not x.empty else np.nan),
).reset_index()

# Merge aggregates
customer_df = agg_features.merge(time_features, on='CustomerId', how='left')

# Add high-cardinality or categorical from original (e.g., most common ProductCategory)
cat_mode = df.groupby('CustomerId')['ProductCategory'].agg(lambda x: x.mode()[0] if not x.empty else 'unknown').reset_index()
customer_df = customer_df.merge(cat_mode, on='CustomerId', how='left')

# 3. Pipeline for preprocessing
numerical_cols = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount',
                  'total_value', 'avg_value', 'transaction_hour_mean', 'transaction_day_mean']

categorical_cols = ['transaction_month', 'ProductCategory']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=10))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit and transform
processed_features = preprocessor.fit_transform(customer_df)

# Save processed features and pipeline
processed_df = pd.DataFrame(processed_features.toarray() if hasattr(processed_features, "toarray") else processed_features,
                            columns=preprocessor.get_feature_names_out())
processed_df['CustomerId'] = customer_df['CustomerId'].values

processed_df.to_csv('../data/processed/processed_features.csv', index=False)

with open('../data/processed/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("Feature engineering complete. Processed data saved.")