import utils
import pickle
import pathlib
import sklearn
import numpy as np
import pandas as pd
from metrics import Feature
from lime import lime_tabular
import matplotlib.pyplot as plt
from joblib import parallel_backend
from typing import Dict, Tuple, Any, List


class ModelRunner:
    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        is_grid_search: bool = False,
        prediction_col: str = "views_per_day",
    ) -> None:
        self.model = model
        self.is_fit = False
        self.prediction_col = prediction_col
        self.is_grid_search = is_grid_search

    def __check_fit(self):
        if not self.is_fit:
            raise NotImplementedError("Model must be fit before it can be trained.")

    def __get_feature_arr(
        self, df: pd.DataFrame, disp_warning=False
    ) -> Tuple[np.ndarray, List[str]]:
        self.__check_fit()
        return utils.get_training_nparray(df, self.feature_tup, disp_warning)

    def __get_y(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(df[self.prediction_col]).astype(float)

    def fit(self, df: pd.DataFrame, feature_tup: Tuple[Feature], threads: int=4) -> None:
        self.is_fit = True
        self.feature_tup = feature_tup
        X, self.feature_names = self.__get_feature_arr(df, disp_warning=True)
        y = self.__get_y(df)
        with parallel_backend('threading', n_jobs=threads):
            self.model.fit(X, y)

    @staticmethod
    def get_corr_matrix(df: pd.DataFrame, feature_tup: Tuple[Feature], output_feature: str) -> pd.Series:
        X, feature_names = utils.get_training_nparray(df, feature_tup, True)
        X_df = pd.DataFrame(X, columns=feature_names)
        return X_df.corrwith(pd.Series(df[output_feature], name=output_feature, dtype=float))
    
    def instance_get_corr_matrix(self, df: pd.DataFrame) -> pd.Series:
        return ModelRunner.get_corr_matrix(df, self.feature_tup, self.prediction_col)

    # Ensure that the testing feature names are identical to training
    def __check_same_feature_names(self, feature_names: List[str]) -> None:
        # I.e. if there are any feature names in the testing data that aren't in the training data
        if set(feature_names) - set(self.feature_names):
            raise AttributeError(
                f"Testing attributes ({feature_names}) don't line up with trained feature names ({self.feature_names})."
            )
        elif set(self.feature_names) - set(feature_names):
            return False
        return True

    # Reorder matrix if feature names are out of order or some needed to be added intothe test matrix
    def __reorder_matrix(
        self, feature_mat: np.ndarray, feature_names: List[str]
    ) -> np.ndarray:
        feature_ls = feature_mat.T.tolist()
        additional_features = list(set(self.feature_names) - set(feature_names))
        num_rows = len(feature_ls[0])
        for additional_feature in additional_features:
            index = self.feature_names.index(additional_feature)
            feature_ls.insert(index, [0] * num_rows)
            feature_names.insert(index, additional_feature)
        if feature_names != self.feature_names:
            raise AttributeError(
                f"Testing attributes ({feature_names}) don't line up with trained feature names ({self.feature_names})."
                "Make sure any One Hot Encoded features are sorted."
            )
        return np.array(feature_ls).T

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, feature_names = self.__get_feature_arr(df)
        if not self.__check_same_feature_names(feature_names):
            X = self.__reorder_matrix(X, feature_names)
        return self.model.predict(X)

    def get_score(self, df: pd.DataFrame) -> float:
        X, feature_names = self.__get_feature_arr(df)
        if not self.__check_same_feature_names(feature_names):
            X = self.__reorder_matrix(X, feature_names)
        y = self.__get_y(df)
        return self.model.score(X, y)

    # Get cv results from a grid search
    def get_cv_results(self) -> Any:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.cv_results_

    # Get the best parameters from a grid search
    def get_best_params(self) -> Dict:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_params_

    # Get best cross validation score from a grid search
    def get_best_score(self) -> float:
        assert self.is_grid_search
        self.__check_fit()
        return self.model.best_score_

    def __get_explainer(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rows: Tuple[int],
        num_samples: int,
    ) -> Any:
        train, _ = self.__get_feature_arr(train_df)
        test, _ = self.__get_feature_arr(test_df)
        explainer = lime_tabular.LimeTabularExplainer(
            train, mode="regression", feature_names=list(self.feature_names)
        )
        for row in rows:
            yield explainer.explain_instance(
                test[row],
                self.model.predict,
                num_features=len(self.feature_names),
                num_samples=num_samples,
            )

    # For python scripts
    def explain_py(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rows: Tuple[int] = (0),
        num_samples: int = 10000,
    ) -> None:
        for explainer in self.__get_explainer(train_df, test_df, rows, num_samples):
            explainer.as_pyplot_figure()
            plt.show()

    # For jupyter notebook scripts
    def explain_notebook(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rows: Tuple[int] = (0),
        num_samples: int = 10000,
    ) -> None:
        for index, explainer in enumerate(self.__get_explainer(train_df, test_df, rows, num_samples)):
            print("True Prediction Val:", self.__get_y(test_df.iloc[rows[index]]))
            explainer.show_in_notebook()

    # Save model runner instance as pickle
    def save(self, file: pathlib.Path) -> None:
        abs_path = str(file)
        if not abs_path.endswith(".pickle"):
            abs_path += ".pickle"

        with open(abs_path, "ab") as f:
            pickle.dump(self, f)

# Load pickled model runner
def load_model(file: pathlib.Path) -> ModelRunner:
    return utils.unpickle_file(file)
