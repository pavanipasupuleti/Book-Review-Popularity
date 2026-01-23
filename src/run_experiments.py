# import pandas as pd

# from run_logistic import run_logistic_experiment
# from run_xgboost import run_xgboost_experiment
# from run_neural_net import run_neural_net_experiment

# # Load processed data
# df = pd.read_csv("data/goodreads_reviews_processed.csv")

# results = []

# # -------------------------
# # LOGISTIC REGRESSION
# # 
# # -------------------------
# for subset in ["A", "B", "C"]:
#     metrics = run_logistic_experiment(
#         df,
#         feature_subset=subset,
#         undersample=True,
#         C=1.0
#     )

#     results.append({
#         "Feature Subset": subset,
#         "Model": "Logistic",
#         "Under Sample": "Yes",
#         "Hyperparameters": "C=1.0",
#         **metrics
#     })

# # -------------------------
# # XGBOOST
# # 
# # -------------------------
# metrics = run_xgboost_experiment(df, feature_subset="A")

# results.append({
#     "Feature Subset": "A",
#     "Model": "XGBoost",
#     "Under Sample": "No",
#     "Hyperparameters": "trees=300, depth=4",
#     **metrics
# })

# # -------------------------
# # NEURAL NETWORK
# # 
# # -------------------------
# metrics = run_neural_net_experiment(df)

# results.append({
#     "Feature Subset": "A",
#     "Model": "Neural Net",
#     "Under Sample": "No",
#     "Hyperparameters": "1 hidden layer (32)",
#     **metrics
# })

# # -------------------------
# # SAVE FINAL TABLE
# # -------------------------
# results_df = pd.DataFrame(results)
# results_df.to_csv("results/experiment_results.csv", index=False)

# print(results_df)

# import pandas as pd

# from run_logistic import run_logistic_experiment
# from run_xgboost import run_xgboost_experiment
# from run_neural_net import run_neural_net_experiment

# # ----------------------------------
# # Load processed data
# # ----------------------------------
# df = pd.read_csv("data/goodreads_reviews_processed.csv")

# results = []

# # ----------------------------------
# # FEATURE SUBSETS
# # ----------------------------------
# feature_subsets = ["A", "B", "C"]

# # ----------------------------------
# # LOGISTIC REGRESSION
# # (undersampling ON)
# # ----------------------------------
# for subset in feature_subsets:
#     metrics = run_logistic_experiment(
#         df,
#         feature_subset=subset,
#         undersample=True,
#         C=1.0
#     )

#     results.append({
#         "Feature Subset": subset,
#         "Model": "Logistic",
#         "Under Sample": "Yes",
#         "Hyperparameters": "C=1.0",
#         **metrics
#     })

# # ----------------------------------
# # XGBOOST
# # (undersampling OFF)
# # ----------------------------------
# for subset in feature_subsets:
#     metrics = run_xgboost_experiment(
#         df,
#         feature_subset=subset
#     )

#     results.append({
#         "Feature Subset": subset,
#         "Model": "XGBoost",
#         "Under Sample": "No",
#         "Hyperparameters": "trees=300, depth=4",
#         **metrics
#     })

# # ----------------------------------
# # NEURAL NETWORK
# # (undersampling OFF, Subset A only)
# # ----------------------------------
# metrics = run_neural_net_experiment(df)

# results.append({
#     "Feature Subset": "A",
#     "Model": "Neural Net",
#     "Under Sample": "No",
#     "Hyperparameters": "1 hidden layer (32)",
#     **metrics
# })


# from run_logistic import run_logistic
# from load_data import load_data

# RANDOM_STATE = 42

# def main():
#     train_df, test_df, y_train, y_test = load_data(random_state=RANDOM_STATE)

#     print("\nRunning Logistic Regression (Paper-aligned)")
#     run_logistic(train_df, test_df, y_train, y_test)

# if __name__ == "__main__":
#     main()

# # ----------------------------------
# # SAVE FINAL TABLE
# # ----------------------------------
# results_df = pd.DataFrame(results)
# results_df.to_csv("results/experiment_results.csv", index=False)

# print(results_df)


# # import pandas as pd
# # from pathlib import Path

# # # -------------------------------
# # # Import experiment runners
# # # -------------------------------
# # from run_logistic import run_logistic_experiment
# # from run_xgboost import run_xgboost_experiment
# # from run_neural_net import run_neural_net_experiment

# # # -------------------------------
# # # Import feature utilities (CORRECT)
# # # -------------------------------
# # from features import build_text_features

# # # -------------------------------
# # # Resolve project paths
# # # -------------------------------
# # BASE_DIR = Path(__file__).resolve().parent.parent
# # DATA_PATH = BASE_DIR / "data" / "goodreads_reviews_processed.csv"
# # RESULTS_DIR = BASE_DIR / "results"
# # RESULTS_DIR.mkdir(exist_ok=True)

# # # -------------------------------
# # # Load data
# # # -------------------------------
# # df = pd.read_csv(DATA_PATH)

# # # -------------------------------
# # # Feature engineering
# # # -------------------------------
# # df = build_text_features(df)

# # # -------------------------------
# # # Run experiments
# # # -------------------------------
# # results = []
# # feature_subsets = ["A", "B", "C"]

# # # ---- Logistic Regression ----
# # for subset in feature_subsets:
# #     m = run_logistic_experiment(df, subset)

# #     results.append({
# #         "Feature Subset": subset,
# #         "Model": "Logistic Regression",
# #         "Under Sample": "Class Weight",
# #         "Hyperparameters": f"C={m['C']}",
# #         "accuracy": m["accuracy"],
# #         "sensitivity": m["sensitivity"],
# #         "specificity": m["specificity"],
# #         "auc": m["auc"]
# #     })

# # # ---- XGBoost ----
# # for subset in feature_subsets:
# #     m = run_xgboost_experiment(df, subset)

# #     results.append({
# #         "Feature Subset": subset,
# #         "Model": "XGBoost",
# #         "Under Sample": "No",
# #         "Hyperparameters": "trees=300, depth=4",
# #         "accuracy": m["accuracy"],
# #         "sensitivity": m["sensitivity"],
# #         "specificity": m["specificity"],
# #         "auc": m["auc"]
# #     })

# # # ---- Neural Network (Subset A only) ----
# # m = run_neural_net_experiment(df)

# # results.append({
# #     "Feature Subset": "A",
# #     "Model": "Neural Network",
# #     "Under Sample": "No",
# #     "Hyperparameters": "1 hidden layer (32 units)",
# #     "accuracy": m["accuracy"],
# #     "sensitivity": m["sensitivity"],
# #     "specificity": m["specificity"],
# #     "auc": m["auc"]
# # })

# # # -------------------------------
# # # Save & print results
# # # -------------------------------
# # results_df = pd.DataFrame(results)
# # results_df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)

# # print("\n===== FINAL EXPERIMENT RESULTS =====\n")
# # print(results_df.to_string(index=False))
# # print("\n===================================\n")


import csv
import pandas as pd
from run_logistic import run_logistic_experiment
from run_xgboost import run_xgboost_experiment
from run_neural_net import run_neural_net_experiment
import pandas as pd

df = pd.read_csv("data/goodreads_reviews_processed.csv")

results = []

configs = [
    ("A", "Logistic", "Yes", "C=0.1"),
    ("B", "Logistic", "Yes", "C=0.1"),
    ("C", "Logistic", "Yes", "C=0.1"),
    ("A", "XGBoost", "No", "trees=300, depth=4"),
    ("B", "XGBoost", "No", "trees=300, depth=4"),
    ("C", "XGBoost", "No", "trees=300, depth=4"),
    ("A", "Neural Net", "No", "1 hidden layer (32)")
]

for feature_subset, model, undersample, params in configs:
    if model == "Logistic":
        metrics = run_logistic_experiment(df, feature_subset, undersample, C=0.1)
    elif model == "XGBoost":
        metrics = run_xgboost_experiment(df, feature_subset)
    else:
        metrics = run_neural_net_experiment(df, feature_subset)

    results.append([
        feature_subset,
        model,
        undersample,
        params,
        metrics["accuracy"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["auc"]
    ])

with open("results/experiment_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Feature Subset",
        "Model",
        "Under Sample",
        "Hyperparameters",
        "accuracy",
        "sensitivity",
        "specificity",
        "auc"
    ])
    writer.writerows(results)

print("experiment_results.csv generated successfully")

results_df = pd.DataFrame(
    results,
    columns=[
        "Feature Subset",
        "Model",
        "Under Sample",
        "Hyperparameters",
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "AUC"
    ]
)

print("\n=== Experiment Results  ===")
print(results_df.to_string(index=False))

print(df.columns.tolist())
exit()
