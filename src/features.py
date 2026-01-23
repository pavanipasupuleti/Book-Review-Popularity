
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
