import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from features import add_text_features, get_tfidf_features

# Load processed data
df = pd.read_csv("data/goodreads_reviews_processed.csv")

# ----- Feature Subset C -----
numeric_features = ["num_words", "avg_word_length", "sentiment"]

X_num = df[numeric_features].values
X_text, tfidf_vectorizer = get_tfidf_features(df["review_text"], max_features=5000)

# Scale numeric features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine numeric + text
X = hstack([X_text, X_num_scaled])
y = df["popular"].values

# Train Logistic Regression
model = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000
)

model.fit(X, y)

# Save everything needed for prediction
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and vectorizer saved successfully.")
