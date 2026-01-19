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

    # ONLY numeric features for XGBoost (safe)
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
