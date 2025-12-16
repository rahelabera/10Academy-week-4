from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Load best model from MLflow
model = mlflow.pyfunc.load_model("models:/BestCreditRiskModel/latest")

@app.post("/predict", response_model=PredictionOutput)
def predict(input: dict):
    # Assume input is processed features dict
    df_input = pd.DataFrame([input])
    prob = model.predict_proba(df_input)[0, 1]
    return {"risk_probability": float(prob), "is_high_risk": int(prob > 0.5)}