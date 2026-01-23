# # from sklearn.neural_network import MLPClassifier
# # import warnings
# # from sklearn.exceptions import ConvergenceWarning

# # from features import build_text_features
# # from load_data import load_data
# # from metrics import get_test_metrics

# # warnings.filterwarnings("ignore", category=ConvergenceWarning)

# # RANDOM_STATE = 42


# # def run_neural_net_experiment(df):
# #     # Split data
# #     train_df, test_df, y_train, y_test = load_data(df, RANDOM_STATE)

# #     # Feature engineering (Subset A only)
# #     train_df = build_text_features(train_df)
# #     test_df = build_text_features(test_df)

# #     X_train = train_df[["review_text"]]
# #     X_test = test_df[["review_text"]]

# #     # Model
# #     model = MLPClassifier(
# #         hidden_layer_sizes=(32,),
# #         max_iter=30,
# #         random_state=RANDOM_STATE
# #     )

# #     model.fit(X_train, y_train)

# #     y_pred = model.predict(X_test)
# #     y_prob = model.predict_proba(X_test)[:, 1]

# #     return get_test_metrics(y_test, y_pred, y_prob)





































# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler

# from evaluate import compute_metrics

# RANDOM_STATE = 42


# def run_neural_net_experiment(df):
#     """
#     Shallow neural network (Mac-safe)
#     """

#     y = df["popular"].values
#     X = df[["num_words", "avg_word_length", "sentiment"]].values

#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
#     )

#     model = MLPClassifier(
#         hidden_layer_sizes=(32,),   # one layer
#         activation="relu",
#         solver="adam",
#         batch_size=64,
#         max_iter=200,
#         early_stopping=True,
#         random_state=RANDOM_STATE
#     )

#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     y_prob = model.predict_proba(X_test)[:, 1]

#     return compute_metrics(y_test, y_pred, y_prob)


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from features import build_features
from metrics import compute_metrics

def run_neural_net_experiment(df, feature_subset):
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df,
        df["popular"],
        test_size=0.2,
        stratify=df["popular"],
        random_state=42
    )

    X_train, X_test = build_features(X_train_df, X_test_df, feature_subset)

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob)
