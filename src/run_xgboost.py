# from xgboost import XGBClassifier

# from features import build_text_features
# from load_data import load_data
# from metrics import get_test_metrics

# RANDOM_STATE = 42


# def select_feature_subset(df, subset):
#     """
#     Subset A: text only
#     Subset B: text + basic numeric features
#     Subset C: all features
#     """
#     if subset == "A":
#         return df[["review_text"]]

#     elif subset == "B":
#         return df.drop(columns=["sentiment_score"], errors="ignore")

#     elif subset == "C":
#         return df

#     else:
#         raise ValueError("Invalid feature subset")


# def run_xgboost_experiment(df, feature_subset):
#     # Split data
#     train_df, test_df, y_train, y_test = load_data(df, RANDOM_STATE)

#     # Feature engineering
#     train_df = build_text_features(train_df)
#     test_df = build_text_features(test_df)

#     train_df = select_feature_subset(train_df, feature_subset)
#     test_df = select_feature_subset(test_df, feature_subset)

#     # Model
#     model = XGBClassifier(
#         n_estimators=300,
#         max_depth=4,
#         eval_metric="logloss",
#         random_state=RANDOM_STATE
#     )

#     model.fit(train_df, y_train)

#     y_pred = model.predict(test_df)
#     y_prob = model.predict_proba(test_df)[:, 1]

#     return get_test_metrics(y_test, y_pred, y_prob)
















import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluate import compute_metrics

RANDOM_STATE = 42


def run_xgboost_experiment(df, feature_subset="A", undersample=False):
    """
    Mac-safe XGBoost experiment
    """

    y = df["popular"].values

    
    X = df[["num_words", "avg_word_length", "sentiment"]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=300,          
        max_depth=4,               
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1                   
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob)
