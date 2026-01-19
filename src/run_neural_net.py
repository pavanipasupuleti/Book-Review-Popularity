import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from evaluate import compute_metrics

RANDOM_STATE = 42


def run_neural_net_experiment(df):
    """
    Shallow neural network (Mac-safe)
    """

    y = df["popular"].values
    X = df[["num_words", "avg_word_length", "sentiment"]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = MLPClassifier(
        hidden_layer_sizes=(32,),   # one layer
        activation="relu",
        solver="adam",
        batch_size=64,
        max_iter=200,
        early_stopping=True,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob)
