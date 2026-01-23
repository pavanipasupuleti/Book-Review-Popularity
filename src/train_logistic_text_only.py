import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/goodreads_reviews_processed.csv")

X = df["review_text"]
y = df["popular"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF (TEXT ONLY)
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = tfidf.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save artifacts
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("Text-only Logistic model saved successfully.")
