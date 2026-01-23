import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from features import (
    add_popularity_label,
    add_text_features,
    get_tfidf_features
)
from load_data import load_reviews


if __name__ == "__main__":

    # ==========================
    # 1. Load data
    # ==========================
    df = load_reviews("data/goodreads_reviews_sample.csv")

    # ==========================
    # 2. Labels + text features
    # ==========================
    df = add_popularity_label(df, threshold=1.0)
    df = add_text_features(df)

    # ==========================
    # 3. TF-IDF
    # ==========================
    X_text, tfidf_vectorizer = get_tfidf_features(
        df["review_text"],
        max_features=5000
    )

    y = df["popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.4, random_state=42
    )

    # ==========================
    # Logistic Regression
    # ==========================
    print("\n--- Logistic Regression ---")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))

    # Feature importance
    feature_names = tfidf_vectorizer.get_feature_names_out()
    coef = lr.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "weight": coef
    }).sort_values(by="weight", ascending=False)

    print("\nTop POPULAR words:")
    print(importance.head(5))

    print("\nTop NON-POPULAR words:")
    print(importance.tail(5))

    # ==========================
    # XGBoost
    # ==========================
    print("\n--- XGBoost ---")

    xgb = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss"
    )

    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    print("XGBoost Accuracy:", xgb_acc)

    # ==========================
    # Neural Network
    # ==========================
    print("\n--- Neural Network ---")

    nn = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="relu",
        max_iter=500,
        random_state=42
    )

    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)

    nn_acc = accuracy_score(y_test, y_pred_nn)
    print("Neural Network Accuracy:", nn_acc)

    # ==========================
    # GRAPHS
    # ==========================

    # 1. Class distribution
    df["popular"].value_counts().plot(kind="bar")
    plt.title("Class Distribution (0 = Not Popular, 1 = Popular)")
    plt.xlabel("Class")
    plt.ylabel("Number of Reviews")
    plt.show()

    # 2. Confusion Matrix (Logistic Regression)
    cm = confusion_matrix(y_test, y_pred_lr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Logistic Regression – Confusion Matrix")
    plt.show()

    # 3. Model comparison
    models = ["Logistic Regression", "XGBoost", "Neural Network"]
    accuracies = [
        accuracy_score(y_test, y_pred_lr),
        xgb_acc,
        nn_acc
    ]

    plt.bar(models, accuracies)
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()

    # 4. Top words bar chart
    top_words = importance.head(10)

    plt.barh(top_words["feature"], top_words["weight"])
    plt.gca().invert_yaxis()
    plt.title("Top Words Pushing Reviews to be Popular")
    plt.show()
