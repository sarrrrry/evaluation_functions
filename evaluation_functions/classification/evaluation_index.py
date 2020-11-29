"""
refer to...
https://qiita.com/FukuharaYohei/items/be89a99c53586fa4e2e4
https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
"""

import pandas as pd

__all__ = [
    "false_negative",
    "false_positive",
    "true_negative",
    "true_positive",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f_measuer"
]


def false_positive(confusion_matrix):
    return confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)


def false_negative(confusion_matrix):
    return confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)


def true_positive(confusion_matrix):
    return np.diag(confusion_matrix)


def true_negative(total_sum, FP, FN, TP):
    return total_sum - (FP + FN + TP)


def accuracy(TP, TN, FP, FN):
    """Overall accuracy"""
    return (TP + TN) / (TP + FP + FN + TN)


def precision(TP, FP):
    """Precision or positive predictive value"""
    return (TP) / (TP + FP)


def recall(TP, FN):
    """Sensitivity, hit rate, recall, or true positive rate"""
    return (TP) / (TP + FN)


def specificity(TN, FP):
    """Specificity or true negative rate"""
    return (TN) / (FP + TN)


def f_measuer(precision, recall):
    return (2 * precision * recall) / (precision + recall)


if __name__ == '__main__':
    import numpy as np

    confusion_matrix = np.array([
        [2., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [1., 0., 2., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0.]
    ])
    col = ["aa", "bb", "cc", "dd", "ee"]
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=col, index=col)

    FP = false_positive(confusion_matrix)
    FN = false_negative(confusion_matrix)
    TP = true_positive(confusion_matrix)
    TN = true_negative(confusion_matrix.values.sum(),
                       FP=FP, FN=FN, TP=TP)

    # Overall accuracy
    prec = precision(TP=TP, FP=FP)
    rec = recall(TP=TP, FN=FN)
    print("acc: \n", accuracy(TP=TP, TN=TN, FP=FP, FN=FN))
    print("prec: \n", prec)
    print("recall: \n", rec)
    print("specificity: \n", specificity(TN=TN, FP=FP))
    print("f-measure: \n", f_measuer(precision=prec, recall=rec))
