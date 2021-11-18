import pickle
import pathlib
import numpy as np
import pandas as pd
from typing import Any, Dict, Callable

def unpickle_file(file: pathlib.Path) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)
    
def unpickle_df(file: pathlib.Path) -> pd.DataFrame:
    return pd.DataFrame(unpickle_file(file))

def get_training_nparray(df: pd.DataFrame, training_features: Dict[str, Callable]) -> np.ndarray:    
    def get_prepped_arr(feature: str, handle: Callable):    
        temp_prepped = handle(df[feature])
        non_zero_count = np.count_nonzero(temp_prepped)
        len_arr = len(temp_prepped[0])

        # If prepped feature returns more than one col, it is probably one hot encoded. Fro OHE, we expect
        # many zero elements so we want to remove the warning if one hot encoded. Otherwise we should make this
        # error check
        try:
            temp_prepped[1]
        except IndexError:
            if non_zero_count < 0.9*len_arr:
                print(f"WARNING: the **{feature}** training feature has nonzero results in only "
                    f"{round(100*non_zero_count/len_arr, 2)}% of data points. Are you sure you want to use it?")
        
        return temp_prepped
    
    return np.array([inner_col for feature, handle in training_features.items()
                        for inner_col in get_prepped_arr(feature, handle)]).T