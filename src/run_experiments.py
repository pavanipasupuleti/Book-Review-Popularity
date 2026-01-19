import pandas as pd

from run_logistic import run_logistic_experiment
from run_xgboost import run_xgboost_experiment
from run_neural_net import run_neural_net_experiment

df = pd.read_csv("data/goodreads_reviews_processed.csv")

results = []

# ---------------- LOGISTIC ----------------
metrics = run_logistic_experiment(df, feature_subset="A", undersample=True, C=1.0)
results.append({
    "Feature Subset": "A",
    "Model": "Logistic",
    "Under Sample": "Yes",
    "Hyperparameters": "C=1.0",
    **metrics
})

# ---------------- XGBOOST ----------------
metrics = run_xgboost_experiment(df, feature_subset="A")
results.append({
    "Feature Subset": "A",
    "Model": "XGBoost",
    "Under Sample": "No",
    "Hyperparameters": "trees=300, depth=4",
    **metrics
})

# ---------------- NEURAL NET ----------------
metrics = run_neural_net_experiment(df)
results.append({
    "Feature Subset": "A",
    "Model": "Neural Net",
    "Under Sample": "No",
    "Hyperparameters": "1 hidden layer (32)",
    **metrics
})

results_df = pd.DataFrame(results)
results_df.to_csv("results/experiment_results.csv", index=False)

print(results_df)
