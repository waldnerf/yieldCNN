import numpy as np
from tensorflow.keras.utils import to_categorical


# -----------------------------------------------------------------------
def add_one_hot(regions):
    """
        Add one hot encoded region information
        INPUT:
            -regions: region ID
        OUTPUT:
            -new_X: corresponding one hot encoded values
    """
    new_X = np.zeros_like(regions)
    cnt = 0
    for r in np.unique(regions):
        new_X[regions == r] = cnt
        cnt += 1
    new_X = to_categorical(new_X, dtype="uint8")
    return new_X

