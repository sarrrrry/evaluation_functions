from typing import Union, Optional, List
import numpy as np
import pandas as pd
import torch

from evaluation_functions.classification.confusion_matrix import confusion_matrix
from evaluation_functions.classification.evaluation_index import *


class ClassificationEvaluator:
    def __call__(self,
                 y_true: Union[np.array, torch.Tensor, List],
                 y_pred: Union[np.array, torch.Tensor, List],
                 N_classes: Optional[int] = None,
                 columns: Optional[list] = None):
        cmx = confusion_matrix(y_true=y_true, y_pred=y_pred, N_classes=N_classes)

        if not columns:
            columns = list(range(N_classes))

        cmx = pd.DataFrame(cmx, columns=columns, index=columns)

        FP = false_positive(cmx)
        FN = false_negative(cmx)
        TP = true_positive(cmx)
        TN = true_negative(cmx.values.sum(),
                           FP=FP, FN=FN, TP=TP)

        # Overall accuracy
        prec = precision(TP=TP, FP=FP)
        rec = recall(TP=TP, FN=FN)
        acc = accuracy(TP=TP, TN=TN, FP=FP, FN=FN)
        spe = specificity(TN=TN, FP=FP)
        f_m = f_measuer(precision=prec, recall=rec)
        df = pd.DataFrame({
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
            "specificity": spe,
            "f_measuer": f_m,
        })
        print(df)


if __name__ == '__main__':
    y_pred = np.ones((100,))
    y_true = np.ones((100,))
    ev = ClassificationEvaluator()
    ev(y_true=y_true, y_pred=y_pred, N_classes=2)
    pass
