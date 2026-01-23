# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import hstack

# from features import build_text_features
# from load_data import load_data
# from metrics import get_test_metrics

# RANDOM_STATE = 42


# def select_feature_subset(df, subset):
#     """
#     A: text only
#     B: text + basic numeric features
#     C: all numeric + text features
#     """
#     if subset == "A":
#         return df[["review_text"]]

#     elif subset == "B":
#         return df.drop(columns=["sentiment_score"], errors="ignore")

#     elif subset == "C":
#         return df

#     else:
#         raise ValueError("Invalid feature subset")


# def run_logistic_experiment(df, feature_subset):
#     # -----------------------------
#     # Train / test split
#     # -----------------------------
#     train_df, test_df, y_train, y_test = load_data(df, RANDOM_STATE)

#     # -----------------------------
#     # Feature engineering
#     # -----------------------------
#     train_df = build_text_features(train_df)
#     test_df = build_text_features(test_df)

#     train_df = select_feature_subset(train_df, feature_subset)
#     test_df = select_feature_subset(test_df, feature_subset)

#     # -----------------------------
#     # Separate TEXT
#     # -----------------------------
#     X_train_text = train_df["review_text"]
#     X_test_text = test_df["review_text"]

#     # -----------------------------
#     # Separate NUMERIC (STRICT)
#     # -----------------------------
#     X_train_num = (
#         train_df.drop(columns=["review_text"], errors="ignore")
#         .select_dtypes(include=["int64", "float64"])
#     )

#     X_test_num = (
#         test_df.drop(columns=["review_text"], errors="ignore")
#         .select_dtypes(include=["int64", "float64"])
#     )

#     # -----------------------------
#     # TF-IDF Vectorization
#     # -----------------------------
#     tfidf = TfidfVectorizer(
#         max_features=5000,
#         stop_words="english"
#     )

#     X_train_text_vec = tfidf.fit_transform(X_train_text)
#     X_test_text_vec = tfidf.transform(X_test_text)

#     # -----------------------------
#     # Combine features
#     # -----------------------------
#     if not X_train_num.empty:
#         scaler = StandardScaler()
#         X_train_num_scaled = scaler.fit_transform(X_train_num)
#         X_test_num_scaled = scaler.transform(X_test_num)

#         X_train_final = hstack([X_train_text_vec, X_train_num_scaled])
#         X_test_final = hstack([X_test_text_vec, X_test_num_scaled])
#     else:
#         X_train_final = X_train_text_vec
#         X_test_final = X_test_text_vec

#     # -----------------------------
#     # Logistic Regression
#     # -----------------------------
#     model = LogisticRegression(
#         C=1.0,
#         max_iter=1000,
#         class_weight="balanced",
#         random_state=RANDOM_STATE
#     )

#     model.fit(X_train_final, y_train)

#     y_pred = model.predict(X_test_final)
#     y_prob = model.predict_proba(X_test_final)[:, 1]

#     return {
#         **get_test_metrics(y_test, y_pred, y_prob),
#         "C": model.C
#     }



























from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from features import build_features
from metrics import compute_metrics

def run_logistic_experiment(df, feature_subset, undersample, C):
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df,
        df["popular"],
        test_size=0.2,
        stratify=df["popular"],
        random_state=42
    )

    X_train, X_test = build_features(X_train_df, X_test_df, feature_subset)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob)







































# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.preprocessing import StandardScaler
# # from scipy.sparse import hstack

# # from evaluate import compute_metrics
# # from features import get_tfidf_features, add_text_features


# # RANDOM_STATE = 42


# # def run_logistic_experiment(
# #     df,
# #     feature_subset="A",
# #     undersample=False,
# #     C=1.0
# # ):
# #     """
# #     Runs ONE experiment row (like one row in the paper table)
# #     """

# #     # Labels
# #     y = df["popular"].values

# #     # Feature subsets
# #     numeric_features = ["num_words", "avg_word_length", "sentiment"]

# #     if feature_subset == "A":
# #         X_num = df[numeric_features]
# #         X_text = None
# #     elif feature_subset == "B":
# #         X_text, _ = get_tfidf_features(df["review_text"])
# #         X_num = None
# #     elif feature_subset == "C":
# #         X_text, _ = get_tfidf_features(df["review_text"])
# #         X_num = df[numeric_features]
# #     else:
# #         raise ValueError("Invalid feature subset")

# #     # Scale numeric features
# #     if X_num is not None:
# #         scaler = StandardScaler()
# #         X_num = scaler.fit_transform(X_num)

# #     # Combine features
# #     if X_text is not None and X_num is not None:
# #         X = hstack([X_text, X_num])
# #     elif X_text is not None:
# #         X = X_text
# #     else:
# #         X = X_num

# #     # Train / Val / Test split
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
# #     )

# #     # Logistic Regression
# #     model = LogisticRegression(
# #         C=C,
# #         penalty="l2",
# #         solver="liblinear",
# #         class_weight="balanced" if undersample else None,
# #         random_state=RANDOM_STATE
# #     )

# #     model.fit(X_train, y_train)

# #     y_pred = model.predict(X_test)
# #     y_prob = model.predict_proba(X_test)[:, 1]

# #     metrics = compute_metrics(y_test, y_pred, y_prob)

# #     return metrics
