# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score

# def get_test_metrics(y_true, y_pred, y_prob):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     return {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
#         "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
#         "auc": roc_auc_score(y_true, y_prob),
        
#         # kept for discussion 
#         "macro_f1": f1_score(y_true, y_pred, average="macro"),
#         "confusion_matrix": [[tn, fp], [fn, tp]]
#     }


from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": tn / (tn + fp),
        "auc": roc_auc_score(y_true, y_prob)
    }
