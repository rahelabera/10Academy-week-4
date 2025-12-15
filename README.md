**Project Overview**
This repository contains code for a credit risk scoring prototype built with alternative behavioral data. The project includes data processing, model training, prediction, and a simple API for serving results. It's structured to be easy to run locally and extend for production.

**Credit Scoring Business Understanding**

**How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**
The Basel II Capital Accord, established by the Basel Committee on Banking Supervision, introduces a risk-sensitive framework for calculating minimum capital requirements through three pillars: 
- Pillar 1 (minimum capital for credit, operational, and market risks),
- Pillar 2 (supervisory review to ensure capital adequacy), and 
- Pillar 3 (market discipline via transparency). 
Its focus on accurate risk measurement, such as probability of default (PD), loss given default (LGD), and exposure at default (EAD), requires banks to use internal ratings-based (IRB) approaches where models must be robust, validated, and auditable. This influences our model by necessitating high interpretability—to allow regulators and risk officers to understand decision-making processes—and thorough documentation for compliance, validation (e.g., via ROC curves and backtesting), and governance. Without these, models risk non-approval or increased capital charges, as seen in guidelines like SR 11-7 from the U.S. Federal Reserve, which emphasize lifecycle oversight to mitigate model risk.

**Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**
Without a direct "default" label in the dataset, we cannot train supervised models on historical loan performance, making it impossible to directly estimate default probabilities. A proxy variable—such as a binary "is_high_risk" label derived from RFM (Recency, Frequency, Monetary) clustering of customer behavior—is necessary to approximate credit risk using available alternative data like transaction patterns. This enables segmentation of customers into high-risk (e.g., disengaged, low-frequency users) and low-risk groups, serving as a surrogate target for model training and allowing predictions in underserved markets like buy-now-pay-later services.
However, relying on proxies introduces business risks, including inaccurate risk assessments if the proxy poorly correlates with actual defaults (e.g., misclassifying stable but low-activity customers as high-risk, leading to lost revenue from denied loans). Other risks include algorithmic bias (e.g., proxies inadvertently correlating with protected demographics like race or gender, causing disparate impact and fair lending violations under regulations like the Equal Credit Opportunity Act), data privacy breaches (e.g., from behavioral data usage under GDPR/PDPO), model unreliability due to overfitting or incomplete data aggregation, understated disparities in predictions, and reputational/legal harm from unfair outcomes. Continuous validation and monitoring are essential to mitigate these.

**What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**
In regulated financial contexts like those governed by Basel II/III, simple models such as Logistic Regression with Weight of Evidence (WoE) offer high interpretability, making it easy to explain coefficients, validate against regulatory standards, and audit for compliance (e.g., understanding how features like transaction frequency influence risk scores). They reduce model risk from overfitting, are computationally efficient, and align with requirements for transparency and fairness, but may sacrifice accuracy on nonlinear or high-dimensional data, potentially leading to suboptimal risk predictions and higher capital reserves.
Complex models like Gradient Boosting (e.g., XGBoost) provide superior performance by capturing intricate patterns in alternative data, improving predictive power (e.g., higher AUC scores) and enabling better financial inclusion for thin-file customers. However, they are often "black-box" in nature, complicating interpretability, increasing validation challenges, and amplifying risks like bias propagation or overfitting, which can violate regulations emphasizing explainability (e.g., EBA guidelines on model governance). Trade-offs include balancing accuracy against regulatory scrutiny—complex models may face higher barriers to approval and require post-hoc tools like SHAP for explanations—while simple models ensure easier deployment but risk underperformance in dynamic markets.

**Installation**
Clone the repo and install dependencies:

```powershell
git clone https://github.com/rahelabera/10Academy-week-4.git
cd 10Academy-week-4
pip install -r requirements.txt
```

**Usage**
Run the main scripts locally:

```powershell
python src/data_processing.py
python src/train.py
python src/predict.py
# Start the API (local dev)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Notes**
- Data folders: `data/raw` (input), `data/processed` (outputs).
- Adjust paths in `src/data_processing.py` if your data is in a different location.
- Run tests with: `pytest -q` (if `pytest` is installed).