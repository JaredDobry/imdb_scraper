import utils
import pickle
import pathlib
import sklearn
import numpy as np
import pandas as pd
from lime import lime_tabular
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List

class ModelRunner:
    def __init__(self, model: sklearn.base.BaseEstimator, is_grid_search: bool=False, prediction_col: str="views_per_day") -> None:
        self.model = model
        self.is_fit = False
        self.prediction_col = prediction_col
        self.is_grid_search = is_grid_search
       
    def __check_fit(self):
        if not self.is_fit:
            raise NotImplementedError("Model must be fit before it can be trained.")
       
    def __get_feature_arr(self, df: pd.DataFrame, disp_warning=False) -> Tuple[np.ndarray, List[str]]:
        self.__check_fit()
        return utils.get_training_nparray(df, self.feature_tup, disp_warning)
    
    def __get_y(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.prediction_col]
        
    def fit(self, df: pd.DataFrame, feature_tup: Tuple[utils.Feature]) -> None:
        self.is_fit = True
        self.feature_tup = feature_tup
        X, self.feature_names = self.__get_feature_arr(df, disp_warning=True)
        y = self.__get_y(df)
        self.model.fit(X, y)
        
    def __check_same_feature_names(self, feature_names: List[str]) -> None:
        # I.e. if there are any feature names in the testing data that aren't in the training data
        if set(feature_names) - set(self.feature_names):
            raise AttributeError(f"Testing attributes ({feature_names}) don't line up with trained feature names ({self.feature_names}).")
        elif set(self.feature_names) - set(feature_names):
            return False
        return True
                
    def __reorder_matrix(self, feature_mat: np.ndarray, feature_names: List[str]):
        feature_ls = feature_mat.T.tolist()
        additional_features = list(set(self.feature_names) - set(feature_names))
        num_rows = len(feature_ls[0])
        for additional_feature in additional_features:
            index = self.feature_names.index(additional_feature)
            feature_ls.insert(index, [0] * num_rows)
        return np.array(feature_ls).T
            
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, feature_names = self.__get_feature_arr(df)
        if not self.__check_same_feature_names(feature_names):
            X = self.__reorder_matrix(X, feature_names)
        return self.model.predict(X)
    
    def get_score(self, df: pd.DataFrame) -> float:
        X = self.__get_feature_arr(df)
        y = self.__get_y(df)
        return self.model.score(X, y)
    
    def get_cv_results(self) -> Any:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.cv_results_
    
    def get_best_params(self) -> Dict:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_params_
    
    def get_best_score(self) -> float:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_score_
    
    def __get_explainer(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rows: Tuple[int], num_samples: int) -> Any:
        train = self.__get_feature_arr(train_df) 
        test = self.__get_feature_arr(test_df)
        explainer = lime_tabular.LimeTabularExplainer(train, mode='regression', feature_names=self.feature_names)
        for row in rows:
            yield explainer.explain_instance(test[row], self.model.predict, num_features=len(self.feature_names), num_samples=num_samples)
    
    def explain_py(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rows: Tuple[int]=(0), num_samples: int=10000) -> None:        
        for explainer in self.__get_explainer(train_df, test_df, rows, num_samples):
            explainer.as_pyplot_figure()
            plt.show()
        
    def explain_notebook(self, train_df: pd.DataFrame, test_df: pd.DataFrame, rows: Tuple[int]=(0), num_samples: int=10000):
        for explainer in self.__get_explainer(train_df, test_df, rows, num_samples):
            explainer.show_in_notebook()
        
    def save(self, file: pathlib.Path) -> None:        
        abs_path = str(file)
        if not abs_path.endswith(".pickle"):
            abs_path += ".pickle"
        
        with open(abs_path, "ab") as f:
            pickle.dump(self, f)

def load_model(file: pathlib.Path) -> ModelRunner:
    return utils.unpickle_file(file)