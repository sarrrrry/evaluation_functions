from typing import Optional

import numpy as np
import torch


# weird trick with bincount
def confusion_matrix(y_true, y_pred, N_classes: Optional[int] = None):
    # calculate Num Of Classes when N_class is None.
    if not N_classes:
        N_classes = max(max(y_true), max(y_pred)) + 1

    if type(y_true) != type(y_pred):
        raise TypeError("'true' and 'pred' have different type. Must be the same type.")

    if isinstance(y_true, torch.Tensor):
        array = lambda x: x.clone().detach().long()
        expand = lambda x: torch.cat([x, torch.zeros(N_classes * N_classes - len(x), dtype=torch.long)])
        bincount = lambda x: torch.bincount(x)
    elif isinstance(y_true, np.ndarray) or isinstance(y_true, list):
        array = lambda x: np.array(x)
        expand = lambda x: np.concatenate([x, np.zeros(N_classes * N_classes - len(y))])
        bincount = lambda x: np.bincount(x)
    else:
        raise TypeError(
            "Type of args is...\n\tExpected: 'list' or 'np.ndarray' or 'torch.Tensor'\n\tActually: {type(y_true)}"
        )

    y_true = array(y_true)
    y_pred = array(y_pred)
    y = N_classes * y_true + y_pred
    y = bincount(y)
    if len(y) < N_classes * N_classes:
        y = expand(y)
    y = y.reshape(N_classes, N_classes)
    return y


if __name__ == '__main__':
    print("from list...")
    y_true = [2, 0, 2, 2, 0, 3]
    y_pred = [0, 0, 2, 2, 0, 2]
    ret = confusion_matrix(y_true, y_pred, 5)
    print(ret)

    print("from np.array...")
    y_true = np.array([2, 0, 2, 2, 0, 3])
    y_pred = np.array([0, 0, 2, 2, 0, 2])
    ret = confusion_matrix(y_true, y_pred, 5)
    print(ret)

    print("from torch.tensor...")
    y_true = torch.Tensor([2, 0, 2, 2, 0, 3])
    y_pred = torch.Tensor([0, 0, 2, 2, 0, 2])
    ret = confusion_matrix(y_true, y_pred, 5)
    print(ret)