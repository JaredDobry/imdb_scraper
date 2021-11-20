import pickle
import pathlib
import numpy as np
import pandas as pd
from numbers import Real
from collections import namedtuple
from typing import Any, List, Tuple
from metrics import Category

Feature = namedtuple('Feature', ["feature_keys", "handle"])

def unpickle_file(file: pathlib.Path) -> Any:
    abs_path = str(file)
    if not abs_path.endswith(".pickle"):
        abs_path += ".pickle"
    with open(abs_path, 'rb') as f:
        return pickle.load(f)
    
def unpickle_df(file: pathlib.Path) -> pd.DataFrame:
    return pd.DataFrame(unpickle_file(file))

def get_training_nparray(df: pd.DataFrame, training_features: Tuple[Feature], disp_warning=False) -> Tuple[np.ndarray, List[str]]:    
    def get_prepped_arr(feature: Feature) -> Category:
        handle = feature.handle
        feature_keys = feature.feature_keys  
        category = handle(df[[feature for feature in feature_keys]], feature_keys)

        # If prepped feature returns more than one col, it is probably one hot encoded. Fro OHE, we expect
        # many zero elements so we want to remove the warning if one hot encoded. Otherwise we should make this
        # error check
        if disp_warning and len(category.category_vals) == 1:
            non_zero_count = np.count_nonzero(category.category_vals)
            len_arr = len(category.category_vals[0])
            if non_zero_count < 0.9*len_arr:
                print(f"WARNING: the **{feature_keys}** training feature has nonzero results in only "
                    f"{round(100*non_zero_count/len_arr, 2)}% of data points. Are you sure you want to use it?")
        return category
    
    training_names = []
    training_ls = []
    for feature_tup in training_features:
        vals, names = get_prepped_arr(feature_tup)
        training_ls.extend(vals)
        training_names.extend(names)
    
    assert len(training_ls) == len(training_names)
    return np.array(training_ls).T, training_names