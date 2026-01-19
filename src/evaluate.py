import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def compute_metrics(y_true, y_pred, y_prob):
    """
    Returns Accuracy, Sensitivity, Specificity, AUC
    """

    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TNR

    auc = roc_auc_score(y_true, y_prob)

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc
    }


def compute_roc(y_true, y_prob):
    """
    Returns ROC curve points
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds
