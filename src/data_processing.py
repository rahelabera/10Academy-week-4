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


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# RFM Calculation
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerId').agg(
    Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
    Frequency=('TransactionId', 'count'),
    Monetary=('Amount', 'sum')  # Use Amount (can be negative for credits)
).reset_index()

# Scale RFM
scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMeans clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze clusters to find high-risk (typically high Recency, low Frequency, low Monetary)
cluster_summary = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("Cluster Summary:\n", cluster_summary)

# Assume cluster with highest Recency & lowest Frequency/Monetary is high-risk (adjust based on print)
high_risk_cluster = cluster_summary.sort_values(['Recency', 'Frequency'], ascending=[False, True]).index[0]

rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# Merge target back
customer_df = customer_df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Save full dataset with target
customer_df.to_csv('../data/processed/customer_with_target.csv', index=False)

with open('../data/processed/rfm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_rfm, f)
with open('../data/processed/kmeans.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

print("Proxy target created and saved.")