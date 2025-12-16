import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# Load processed data with target
df = pd.read_csv('../data/processed/customer_with_target.csv')
with open('../data/processed/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

features = preprocessor.transform(df.drop(columns=['CustomerId', 'is_high_risk']))
target = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42)
}

params = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, None]}
}

mlflow.set_experiment("credit_risk_model")

best_auc = 0
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        grid = GridSearchCV(model, params[name], cv=3, scoring='roc_auc')
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        probs = grid.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds),
            'recall': recall_score(y_test, preds),
            'f1': f1_score(y_test, preds),
            'auc': roc_auc_score(y_test, probs)
        }
        
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(grid.best_estimator_, "model")
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_model = grid.best_estimator_
        
        print(f"{name} AUC: {metrics['auc']:.4f}")

# Register best model
model_uri = mlflow.get_artifact_uri("model")
mlflow.register_model(model_uri, "BestCreditRiskModel")

# Save best locally
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Training complete. Best model registered in MLflow.")