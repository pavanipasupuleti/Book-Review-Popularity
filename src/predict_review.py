# import joblib
# import pandas as pd
# from scipy.sparse import hstack

# from features import add_text_features

# # -----------------------------
# # Load trained components
# # -----------------------------
# model = joblib.load("models/logistic_model.pkl")
# tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
# scaler = joblib.load("models/scaler.pkl")

# # -----------------------------
# # Take user input
# # -----------------------------
# print("\nEnter a review (press Enter twice to finish):")

# lines = []
# while True:
#     line = input()
#     if line.strip() == "":
#         break
#     lines.append(line)

# review_text = " ".join(lines)

# if len(review_text.strip()) == 0:
#     print("No review entered. Exiting.")
#     exit()

# # -----------------------------
# # Convert to DataFrame
# # -----------------------------
# df = pd.DataFrame({"review_text": [review_text]})

# # -----------------------------
# # Feature engineering
# # -----------------------------
# df = add_text_features(df)
# numeric_features = ["num_words", "avg_word_length", "sentiment"]

# X_num = df[numeric_features].values
# X_num_scaled = scaler.transform(X_num)

# # TF-IDF features
# X_text = tfidf_vectorizer.transform(df["review_text"])

# # Combine features (Subset C)
# X = hstack([X_text, X_num_scaled])

# # -----------------------------
# # Predict
# # -----------------------------
# prob = model.predict_proba(X)[0][1]

# THRESHOLD = 0.3  # decision threshold for demo
# prediction = "POPULAR" if prob >= THRESHOLD else "NOT POPULAR"

# # -----------------------------
# # Output
# # -----------------------------
# print("\n-----------------------------")
# # print(f"Predicted probability: {prob:.2f}")
# print(f"Prediction: {prediction}")
# print("-----------------------------\n")

# import joblib
# import pandas as pd

# def main():
#     # -----------------------------
#     # Load trained components
#     # -----------------------------
#     model = joblib.load("models/logistic_model.pkl")
#     tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

#     # -----------------------------
#     # Take user input
#     # -----------------------------
#     print("\nEnter the review text.")
#     print("Type END on a new line when finished.\n")

#     lines = []
#     while True:
#         line = input()
#         if line.strip().upper() == "END":
#             break
#         lines.append(line)

#     review_text = " ".join(lines).strip()

#     if not review_text:
#         print("No review entered. Exiting.")
#         return

#     # -----------------------------
#     # Vectorize text (SAME AS TRAINING)
#     # -----------------------------
#     X = tfidf_vectorizer.transform([review_text])

#     # -----------------------------
#     # Predict
#     # -----------------------------
#     prob = model.predict_proba(X)[0][1]

#     THRESHOLD = 0.3
#     prediction = "POPULAR" if prob >= THRESHOLD else "NOT POPULAR"

#     # -----------------------------
#     # Output
#     # -----------------------------
#     print("\n-----------------------------")
#     print(f"Prediction: {prediction}")
   
#     print("-----------------------------\n")


# if __name__ == "__main__":
#     main()

import joblib

def main():
    # -----------------------------
    # Load trained components
    # -----------------------------
    model = joblib.load("models/logistic_model.pkl")
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    # -----------------------------
    # Take user input (ENTER to finish)
    # -----------------------------
    print("\nEnter the review text.")
    print("Press ENTER on an empty line to finish.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    review_text = " ".join(lines).strip()

    if not review_text:
        print("No review entered. Exiting.")
        return

    # -----------------------------
    # Vectorize text (same as training)
    # -----------------------------
    X = tfidf_vectorizer.transform([review_text])

    # -----------------------------
    # Predict
    # -----------------------------
    prob = model.predict_proba(X)[0][1]

    THRESHOLD = 0.3
    prediction = "POPULAR" if prob >= THRESHOLD else "NOT POPULAR"

    # -----------------------------
    # Output
    # -----------------------------
    print("\n-----------------------------")
    print(f"Prediction: {prediction}")
   
    print("-----------------------------\n")


if __name__ == "__main__":
    main()
