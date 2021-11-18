from os import stat
import utils
import pickle
import pathlib
import sklearn
import numpy as np
import pandas as pd
from typing import Tuple


class ModelRunner:
    def __init__(self, model: sklearn.base.BaseEstimator, prediction_col: str="views_per_day") -> None:
        self.model = model
        self.is_fit = False
        self.prediction_col = prediction_col
       
    def __get_feature_arr(self, df: pd.DataFrame, disp_warning=False) -> np.ndarray:
        if not self.is_fit:
            raise NotImplementedError("Model must be fit before it can be trained.")
        return utils.get_training_nparray(df, self.feature_tup, disp_warning)  
    
    def __get_y(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.prediction_col]
        
    def fit(self, df: pd.DataFrame, feature_tup: Tuple[utils.Feature]) -> None:
        self.is_fit = True
        self.feature_tup = feature_tup
        X = self.__get_feature_arr(df, disp_warning=False)
        print(X)
        y = self.__get_y(df)
        self.model.fit(X, y)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.__get_feature_arr(df)
        return self.model.predict(X)
    
    def score(self, df: pd.DataFrame) -> float:
        X = self.__get_feature_arr(df)
        y = self.__get_y(df)
        return self.model.score(X, y)
    
    def save(self, file: pathlib.Path):
        abs_path = str(file)
        if not abs_path.endswith(".pickle"):
            abs_path += ".pickle"
        
        with open(abs_path, "ab") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: pathlib.Path):
        return utils.unpickle_file(file)