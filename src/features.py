import nltk
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


def add_popularity_label(df, threshold=1.0):
    """
    Adds a binary popularity label based on engagement.
    popular = 1 if (n_votes + n_comments) >= threshold else 0
    """

    df = df.copy()

    df["total_engagement"] = df["n_votes"] + df["n_comments"]
    df["popular"] = (df["total_engagement"] >= threshold).astype(int)

    print(f"Popularity threshold (>=): {threshold}")
    print(df["popular"].value_counts())

    return df


def add_text_features(df):
    """
    Add simple text-based features:
    - number of words
    - average word length
    - sentiment score
    """

    df = df.copy()

    df["num_words"] = df["review_text"].astype(str).apply(
        lambda x: len(x.split())
    )

    df["avg_word_length"] = df["review_text"].astype(str).apply(
        lambda x: sum(len(w) for w in x.split()) / max(len(x.split()), 1)
    )

    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

    df["sentiment"] = df["review_text"].astype(str).apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    return df


def get_tfidf_features(text_series, max_features=5000):
    """
    Convert review text into TF-IDF features
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english"
    )

    X = vectorizer.fit_transform(text_series.astype(str))

    return X, vectorizer




































# import pandas as pd

# def add_popularity_label(df, threshold=0.5):

#     """
#     Adds two columns:
#     1. total_engagement = likes + comments
#     2. popular = 1 if likes_share > threshold else 0
#     """

#     # total likes + comments for each review
#     df["total_engagement"] = df["likes"] + df["comments"]

#     # total engagement per book
#     book_totals = df.groupby("book_id")["total_engagement"].transform("sum")

#     # share of engagement
#     df["likes_share"] = df["total_engagement"] / book_totals

#     # popularity label
#     df["popular"] = (df["likes_share"] > threshold).astype(int)

#     return df


# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer

# # download required data once
# nltk.download("vader_lexicon")

# def add_text_features(df):
#     """
#     Adds basic text features:
#     - num_words
#     - avg_word_length
#     - sentiment score
#     """

#     sia = SentimentIntensityAnalyzer()

#     # number of words in review
#     df["num_words"] = df["review_text"].apply(
#         lambda x: len(x.split())
#     )

#     # average word length
#     df["avg_word_length"] = df["review_text"].apply(
#         lambda x: sum(len(word) for word in x.split()) / len(x.split())
#     )

#     # sentiment score (-1 = negative, +1 = positive)
#     df["sentiment"] = df["review_text"].apply(
#         lambda x: sia.polarity_scores(x)["compound"]
#     )

#     return df


# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# def get_bow_features(text_series, max_features=20):
#     """
#     Converts text into Bag of Words features
#     """
#     bow = CountVectorizer(
#         stop_words="english",
#         max_features=max_features
#     )
#     bow_matrix = bow.fit_transform(text_series)
#     return bow_matrix, bow.get_feature_names_out()


# def get_tfidf_features(text_series, max_features=20):
#     """
#     Converts text into TF-IDF features
#     """
#     tfidf = TfidfVectorizer(
#         stop_words="english",
#         max_features=max_features
#     )
#     tfidf_matrix = tfidf.fit_transform(text_series)
#     return tfidf_matrix, tfidf.get_feature_names_out()
