import utils
import pickle
import pathlib
import sklearn
import numpy as np
import pandas as pd
from typing import Tuple, Any


class ModelRunner:
    def __init__(self, model: sklearn.base.BaseEstimator, is_grid_search: bool=False, prediction_col: str="views_per_day") -> None:
        self.model = model
        self.is_fit = False
        self.prediction_col = prediction_col
        self.is_grid_search = is_grid_search
       
    def __check_fit(self):
        if not self.is_fit:
            raise NotImplementedError("Model must be fit before it can be trained.")
       
    def __get_feature_arr(self, df: pd.DataFrame, disp_warning=False) -> np.ndarray:
        self.__check_fit()
        return utils.get_training_nparray(df, self.feature_tup, disp_warning)  
    
    def __get_y(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.prediction_col]
        
    def fit(self, df: pd.DataFrame, feature_tup: Tuple[utils.Feature]) -> None:
        self.is_fit = True
        self.feature_tup = feature_tup
        X = self.__get_feature_arr(df, disp_warning=True)
        y = self.__get_y(df)
        self.model.fit(X, y)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.__get_feature_arr(df)
        return self.model.predict(X)
    
    def score(self, df: pd.DataFrame) -> float:
        X = self.__get_feature_arr(df)
        y = self.__get_y(df)
        return self.model.score(X, y)
    
    def get_cv_results(self):
        assert self.is_grid_search
        self.__check_fit()
        return self.model.cv_results_
    
    def get_best_params(self):
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_params_
    
    def get_best_score(self):
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_score_
    
    def save(self, file: pathlib.Path):
        abs_path = str(file)
        if not abs_path.endswith(".pickle"):
            abs_path += ".pickle"
        
        with open(abs_path, "ab") as f:
            pickle.dump(self, f)

def load_model(file: pathlib.Path) -> ModelRunner:
    return utils.unpickle_file(file)