from sklearn.model_selection import train_test_split

def load_data(df, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["popular"],
        random_state=random_state
    )
    return (
        train_df,
        test_df,
        train_df["popular"],
        test_df["popular"]
    )















# import pandas as pd

# from features import (
#     add_popularity_label,
#     add_text_features,
#     get_tfidf_features
# )

# def load_reviews(path):
#     return pd.read_csv(path)

# if __name__ == "__main__":
#     reviews = load_reviews("data/goodreads_reviews_sample.csv")

#     reviews = add_popularity_label(reviews)
#     reviews = add_text_features(reviews)

#     X_text, tfidf_vectorizer = get_tfidf_features(
#         reviews["review_text"],
#         max_features=5000
#     )
