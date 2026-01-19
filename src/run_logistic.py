import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from evaluate import compute_metrics
from features import get_tfidf_features, add_text_features


RANDOM_STATE = 42


def run_logistic_experiment(
    df,
    feature_subset="A",
    undersample=False,
    C=1.0
):
    """
    Runs ONE experiment row (like one row in the paper table)
    """

    # Labels
    y = df["popular"].values

    # Feature subsets
    numeric_features = ["num_words", "avg_word_length", "sentiment"]

    if feature_subset == "A":
        X_num = df[numeric_features]
        X_text = None
    elif feature_subset == "B":
        X_text, _ = get_tfidf_features(df["review_text"])
        X_num = None
    elif feature_subset == "C":
        X_text, _ = get_tfidf_features(df["review_text"])
        X_num = df[numeric_features]
    else:
        raise ValueError("Invalid feature subset")

    # Scale numeric features
    if X_num is not None:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)

    # Combine features
    if X_text is not None and X_num is not None:
        X = hstack([X_text, X_num])
    elif X_text is not None:
        X = X_text
    else:
        X = X_num

    # Train / Val / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Logistic Regression
    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced" if undersample else None,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)

    return metrics
