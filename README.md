# Text-Based Prediction of Book Review Popularity

## Overview

Online platforms such as Goodreads host millions of user-generated book reviews, but only a small fraction of these reviews receive significant engagement in the form of votes and comments. Automatically identifying such high-impact reviews can help platforms highlight valuable content and improve user experience.

This project addresses the problem of **predicting the popularity of a book review based solely on its text content and derived features**. The task is formulated as a **binary classification problem** using machine learning techniques.


## Problem Statement

Given a book review from Goodreads, predict whether the review will be **popular** or **not popular** based on textual and engagement-related features.

### Classification Labels

* **Popular (1)** → Review receives high engagement
* **Not Popular (0)** → Review receives low engagement

The objective is to build and evaluate machine learning models that can accurately perform this classification.

---

## Approach

The solution follows a structured machine learning pipeline:

### 1. Data Preparation

* Goodreads review data is collected and converted into CSV format.
* Engagement features such as votes and comments are used to define popularity.

### 2. Feature Engineering

**Text-based features**

* Number of words in the review
* Average word length
* Sentiment score using VADER

**Vector-based features**

* TF-IDF representation of review text

### 3. Label Generation

* Reviews are labeled as *popular* or *not popular* based on a predefined engagement threshold.

### 4. Model Training

The following models are trained and compared:

* Logistic Regression
* XGBoost
* Neural Network (MLP)

### 5. Evaluation

Models are evaluated using:

* Accuracy
* Sensitivity (Recall)
* Specificity
* AUC score

### 6. Prediction

* The final trained model is saved.
* Users can enter new reviews to predict popularity.

---

## Project Structure

```
Book-Review-Popularity/
│
├── data/
│   ├── raw/                    # Raw Goodreads data
│   └── processed/              # Cleaned and labeled datasets
│
├── models/
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── scaler.pkl
│
├── src/
│   ├── prepare_real_data.py
│   ├── preprocess_data.py
│   ├── features.py
│   ├── train.py
│   ├── train_and_save_model.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   └── predict_review.py
│
├── requirements.txt
└── README.md
```

---

## File Description

* **prepare_real_data.py**:
  Converts raw Goodreads JSON data into structured CSV format.

* **preprocess_data.py**:
  Adds popularity labels and basic text-based features.

* **features.py**:
  Implements feature engineering, including:

  * Sentiment analysis
  * TF-IDF vectorization
  * Statistical text features

* **train.py**:
  Trains and compares Logistic Regression, XGBoost, and Neural Network models.

* **train_and_save_model.py**:
  Trains the final Logistic Regression model and saves:

  * Trained model
  * TF-IDF vectorizer
  * Feature scaler

* **evaluate.py**:
  Computes evaluation metrics such as accuracy, sensitivity, specificity, and AUC.

* **run_experiments.py**:
  Runs experiments with different feature subsets and records results.

* **predict_review.py**:
  Accepts a user-entered review and predicts whether it is popular or not.

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/pavanipasupuleti/Book-Review-Popularity.git
cd Book-Review-Popularity
```

### Step 2: Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS 
venv\Scripts\activate           # Windows
```


---

## How to Run the Project

### Step 1: Prepare the Dataset

```bash
python src/prepare_real_data.py
python src/preprocess_data.py
```

### Step 2: Train Models and Run Experiments

```bash
python src/run_experiments.py
```

### Step 3: Train and Save Final Model

```bash
python src/train_and_save_model.py
```

### Step 4: Predict Review Popularity

```bash
python src/predict_review.py
```

Enter a review when prompted.
The system outputs:

* **POPULAR**
* **NOT POPULAR**

---

## Results

* **Logistic Regression (with undersampling)**

  * Accuracy: ~71.5%
  * AUC: ~0.68
  * Low sensitivity (misses many popular reviews)

* **XGBoost**

  * Accuracy: ~70.8%
  * Slightly better sensitivity
  * Comparable overall performance

* **Neural Network**

  * Accuracy: ~68.6%
  * Highest sensitivity
  * More false positives

### Key Observation

* Negative reviews can still be popular.
* Positive sentiment alone does not guarantee popularity.

---

## Challenges Faced

* Severe class imbalance between popular and non-popular reviews
* High dimensionality due to TF-IDF features
* File path handling across operating systems
* Model persistence required explicit directory management

---

## Conclusion

* Text-based features combined with TF-IDF are effective for predicting review popularity.
* Logistic Regression provides strong performance with simplicity and interpretability.
* Review popularity depends more on structure and engagement patterns than sentiment alone.
* The system demonstrates a practical application of machine learning for real-world text analytics.

---------------

## Results


<img width="1600" height="283" alt="image" src="https://github.com/user-attachments/assets/297bba9f-9419-4020-81ed-a0d5b6c27e47" />


-------------



<img width="1560" height="438" alt="image" src="https://github.com/user-attachments/assets/f5475d56-753d-4a14-b90f-082e9645da35" />



---------------





<img width="1600" height="414" alt="image" src="https://github.com/user-attachments/assets/e57eb7dd-2f5f-42bf-abeb-52620cbaaacc" />



--------

