import random
import pickle
import pathlib
import numpy as np
import pandas as pd
from typing import Any, List, Tuple
from metrics import Category, Feature


def unpickle_file(file: pathlib.Path) -> Any:
    abs_path = str(file)
    if not abs_path.endswith(".pickle"):
        abs_path += ".pickle"
    with open(abs_path, "rb") as f:
        return pickle.load(f)


def unpickle_df(file: pathlib.Path) -> pd.DataFrame:
    return pd.DataFrame(unpickle_file(file))


def get_training_nparray(
    df: pd.DataFrame, training_features: Tuple[Feature], disp_warning=False
) -> Tuple[np.ndarray, List[str]]:
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
            if non_zero_count < 0.9 * len_arr:
                print(
                    f"WARNING: the **{feature_keys}** training feature has nonzero results in only "
                    f"{round(100*non_zero_count/len_arr, 2)}% of data points. Are you sure you want to use it?"
                )
        return category

    training_names = []
    training_ls = []
    for feature_tup in training_features:
        vals, names = get_prepped_arr(feature_tup)
        training_ls.extend(vals)
        training_names.extend(names)

    assert len(training_ls) == len(training_names)
    return np.array(training_ls).T, training_names


def rm_rows_missing_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    
    def strikes_lt_n(row: pd.Series) -> bool:
        strikes = 0

        if row["budget"] == "0":
            strikes += 1

        if row["revenue"] == "0":
            strikes += 1

        if ["runtime"] == "0":
            strikes += 1

        if not row["genres"]:
            strikes += 1

        if not row["spoken_languages"]:
            strikes += 1

        if not row["original_language"]:
            strikes += 1

        year = row["release_date"].partition("-")[0]
        if year == "null" or int(year) < 1950:
            strikes +=1
            
        return strikes < n
    
    strikes_lt_n_ls = np.array([strikes_lt_n(row) for _, row in df.iterrows()], dtype=bool)
    return df.loc[strikes_lt_n_ls, :]

def train_test_split(df: pd.DataFrame, test_percent: int, seed: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    len_df = len(df)
    num_test = test_percent*len_df//100
    all_indeces = range(len_df)
    random.seed(seed)
    indeces_test = set(random.sample(all_indeces, k=num_test))
    bool_list_test = [index in indeces_test for index in all_indeces]
    bool_list_train = [not test for test in bool_list_test]
    return df[bool_list_train], df[bool_list_test]
