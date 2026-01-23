# import numpy as np
# from scipy.sparse import hstack
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import StandardScaler

# def add_popularity_label(df):
#     """
#     Dataset already contains the popularity label.
#     This function simply ensures it exists and is binary.
#     """
#     df = df.copy()

#     if "popular" not in df.columns:
#         raise ValueError("Expected 'popular' column not found in dataset")

#     df["popular"] = df["popular"].astype(int)
#     return df

# def add_text_features(df):
#     """
#     Text-derived features already exist in the dataset.
#     This function is kept for pipeline consistency.
#     """
#     required = ["num_words", "avg_word_length", "sentiment"]
#     missing = [c for c in required if c not in df.columns]

#     if missing:
#         raise ValueError(f"Missing required text features: {missing}")

#     return df


# def get_tfidf_features(train_text, test_text, max_features=5000):
#     tfidf = TfidfVectorizer(
#         max_features=max_features,
#         stop_words="english",
#         ngram_range=(1, 2)
#     )
#     X_train_text = tfidf.fit_transform(train_text)
#     X_test_text = tfidf.transform(test_text)
#     return X_train_text, X_test_text

# def build_features(train_df, test_df, feature_subset):
#     X_train_text, X_test_text = get_tfidf_features(
#         train_df["review_text"],
#         test_df["review_text"]
#     )

#     if feature_subset == "A":
#         return X_train_text, X_test_text

#     meta_cols = ["num_words"]

#     if feature_subset == "C":
#         meta_cols.extend(["avg_word_length", "sentiment", "rating"])

#     from sklearn.preprocessing import StandardScaler
#     from scipy.sparse import hstack

#     scaler = StandardScaler()

#     X_train_meta = scaler.fit_transform(train_df[meta_cols])
#     X_test_meta = scaler.transform(test_df[meta_cols])

#     X_train = hstack([X_train_text, X_train_meta])
#     X_test = hstack([X_test_text, X_test_meta])

#     return X_train, X_test


# # import numpy as np

# # # -------------------------------------------------
# # # Sentiment (simple, deterministic, dependency-free)
# # # -------------------------------------------------
# # def get_sentiment_score(text):
# #     if not isinstance(text, str) or text.strip() == "":
# #         return 0.0

# #     positive_words = ["good", "great", "amazing", "love", "excellent", "best"]
# #     negative_words = ["bad", "worst", "boring", "hate", "terrible", "awful"]

# #     text = text.lower()
# #     score = 0

# #     for w in positive_words:
# #         if w in text:
# #             score += 1
# #     for w in negative_words:
# #         if w in text:
# #             score -= 1

# #     return score


# # # -------------------------------------------------
# # # BUILD features (used everywhere)
# # # -------------------------------------------------
# # def build_text_features(df, text_col="review_text"):
# #     if text_col not in df.columns:
# #         raise ValueError(f"Expected column '{text_col}'")

# #     df = df.copy()

# #     df["num_words"] = df[text_col].apply(lambda x: len(x.split()))
# #     df["avg_word_length"] = df[text_col].apply(
# #         lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0.0
# #     )
# #     df["sentiment"] = df[text_col].apply(get_sentiment_score)

# #     return df


# # # -------------------------------------------------
# # # VALIDATE features (used only in experiments)
# # # -------------------------------------------------
# # def validate_text_features(df):
# #     required = ["num_words", "avg_word_length", "sentiment"]
# #     missing = [c for c in required if c not in df.columns]

# #     if missing:
# #         raise ValueError(f"Missing required text features: {missing}")




from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer

def build_features(train_df, test_df, feature_subset):
    """
    Feature Subset:
    A -> Text only
    B -> Text only + basic numeric features
    C -> Text only + all numeric features
    """

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train = tfidf.fit_transform(train_df["review_text"])
    X_test = tfidf.transform(test_df["review_text"])

    return X_train, X_test
