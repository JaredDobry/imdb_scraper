from typing import List
import numpy as np
import pandas as pd

# TODO: @Jared and Hyun, every one of these functions should return a
# List[List[numeric]] -> Since we need one hot encoding, many of these will
# return a two dimensional data collection. Therefore, the get_training_nparray
# function just assumes that it data will always be 2-dimensional

def __get_numeric__(ds: pd.Series) -> List[List[float]]:
    return [[float(x) for x in ds]]

def get_belongs_to_collection(ds: pd.DataFrame) -> List[List[int]]:
    return [[1 if d else 0 for d in ds["belongs_to_collection"]]]

def get_budget(ds: pd.DataFrame) -> List[List[float]]:
    return __get_numeric__(ds["budget"])

# TODO: I'd suggest using the sklearn.preprocessing.one_hot_encoder OR pandas get_dummies
# for it. Remember to return results as List[List[int]] (i.e. 0 or 1)
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def get_genres(ds: pd.DataFrame) -> List[List[int]]:
    # genre_list is stored in a pd.Series -> list -> dict. Actual genre is dict["name"]
    pass    
    
    

