import utils
import metrics
import pathlib
import warnings
import argparse
import numpy as np
from typing import Dict
from datetime import datetime
from model_runner import ModelRunner
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH / "StaticDB"
MODEL_PATH = PARENT_PATH / "models"

parser = argparse.ArgumentParser()
parser.add_argument("regressor", choices=["L", "Linear",
                                          "R", "Ridge",
                                          "DT", "Decision Tree", 
                                          "RF", "Random Forest", 
                                          "SVM", "Linear SVM"],
                    help="Regressor training model")

parser.add_argument("-g", action="store_true", help="If the flag is listed, grid search will be applied.")
parser.add_argument("-s", action="store_true", help="If present, will save model in models folder.")

def print_w_sep(*args) -> None:
    print("\n", 50 * "-", sep="")
    for arg in args:
        print(arg)
    print(50 * "-")

def get_model_dict(regressor_type: str) -> Dict:
    model_dict = {}
    if regressor_type == "L" or regressor_type == "Linear":
        # Linear Regression
        from sklearn.linear_model import LinearRegression
        model_dict["model_type"] = LinearRegression(normalize=True)
        model_dict["empty_model_type"] = LinearRegression
        model_dict["params"] = {"normalize": [True, False]}
        model_dict["model_name"] = "LinearRegression"
    elif regressor_type == "R" or regressor_type == "Ridge":
        # Ridge Regression
        from sklearn.linear_model import Ridge
        model_dict["model_type"] = Ridge(alpha=.14, normalize=True)
        model_dict["empty_model_type"] = Ridge
        model_dict["params"] = {"alpha": np.logspace(-9, 9, num=1000), "normalize": [True, False]}
        model_dict["model_name"] = "RidgeRegression"
    elif regressor_type == "DT" or regressor_type == "Decision Tree":
        # Decision Tree Regressor
        from sklearn.tree import DecisionTreeRegressor
        model_dict["model_type"] = DecisionTreeRegressor(max_depth=8, max_features='auto', max_leaf_nodes=50, min_samples_leaf=5, min_weight_fraction_leaf=0, splitter='best')
        model_dict["params"] = {
            "splitter": ["best"],
            "max_depth": np.linspace(1, 15, 5, dtype=int),
            "min_samples_leaf": np.linspace(1, 10, 5, dtype=int),
            "min_weight_fraction_leaf": np.linspace(0.1, 0.9, 3, dtype=int),
            "max_features": ["auto"],
            "max_leaf_nodes": [None] + list(np.linspace(10, 90, 3, dtype=int))
        }
        model_dict["empty_model_type"] = DecisionTreeRegressor
        model_dict["model_name"] = "DecisionTreeRegressor"
    elif regressor_type == "RF" or regressor_type == "Random Forest":
        # Random Forest Regressor
        from sklearn.ensemble import RandomForestRegressor
        model_dict["model_type"] = RandomForestRegressor(max_depth=8, max_features="auto", max_leaf_nodes=50, min_samples_leaf=20)
        model_dict["params"] = {
            "max_depth": np.linspace(1, 15, 5, dtype=int),                      # max depth of tree
            "min_samples_leaf": np.linspace(1, 20, 5, dtype=int),               # min num of samples to split a node
            "max_features": ["auto"],                                           # number of features when looking for best split
            "max_leaf_nodes": [None] + list(np.linspace(10, 90, 3, dtype=int))  # maximum number of leaf nodes
        }
        model_dict["empty_model_type"] = RandomForestRegressor
        model_dict["model_name"] = "RandomForestRegressor"
    elif regressor_type == "SVM" or regressor_type == "Linear SVM":
        # Support Vector Machine (SVM) Regressor
        from sklearn.svm import LinearSVR
        model_dict["model_type"] = LinearSVR(epsilon=1e-1, tol=0.1, C=2, max_iter=5000)
        model_dict["params"] = {
            "tol": np.linspace(1e-5, 1e-3, 5, dtype=float),
            "epsilon": np.linspace(0.00001, 1e-2, 5, dtype=float),
            "C": np.linspace(1, 10, 3, dtype=int),
            "fit_intercept": [True, False],
            "intercept_scaling": np.linspace(1, 10, 5, dtype=int),    
        }
        model_dict["empty_model_type"] = LinearSVR
        model_dict["model_name"] = "SVMRegressor"
    
    return model_dict

if __name__ == "__main__":
    args = parser.parse_args()
    is_grid_search = args.g
    
    train_df_pickled = utils.unpickle_df(DB_PATH/"train_movies.pickle")
    pre_cleaned_len = len(train_df_pickled)
    train_df_pickled = utils.rm_rows_missing_data(train_df_pickled, 3)
    train_df, validation_df = utils.train_test_split(train_df_pickled, 20, 42)

    data_name = f"noextras_datalen_{len(train_df)}"
    print_w_sep(f"Unpacked {pre_cleaned_len} rows of training data. Cut down dataset to {len(train_df)+len(validation_df)} training rows."
        " 20% will be dedicated to validation.")

    feature_tup = (
        metrics.Feature(("budget",), metrics.get_numeric),                              # has zeros ro remove
        metrics.Feature(("belongs_to_collection",), metrics.get_belongs_to_collection), # not a strike
        metrics.Feature(("genres",), metrics.get_genres),                               # has some empty lists to remove
        metrics.Feature(("original_language",), metrics.get_original_language),         # cannot find if there are emptys
        metrics.Feature(("views_per_day",), metrics.get_numeric),
        metrics.Feature(("release_date",), metrics.get_release_year),                   # at least has has null values, idk if there are zeros
        metrics.Feature(("revenue",), metrics.get_numeric),                             # has zero values
        metrics.Feature(("runtime",), metrics.get_numeric),                             # could have zeros, int64
        metrics.Feature(("spoken_languages",), metrics.get_num_spoken_languages),       # has some empty lists 
        metrics.Feature(("vote_average", "vote_count",), metrics.get_vote_popularity),
    )
    print_w_sep(f"Defined feature tuple. Using features {[keys for feature in feature_tup for keys in feature.feature_keys]}")

    model_dict = get_model_dict(args.regressor)
    
    if is_grid_search:
        # Grid search
        model_dict["model_type"] = GridSearchCV(model_dict["empty_model_type"](),
                                                model_dict["params"], cv=10)
        model_dict["model_name"] = "GridSearch_" + model_dict["model_name"]
    
    model = ModelRunner(model_dict["model_type"], is_grid_search=is_grid_search, prediction_col="popularity")
    if is_grid_search:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_df, feature_tup)
            print_w_sep(f"Best Cross Validation Model Score: {model.get_best_score()}",
                        f"Params for Best Model Score: {model.get_best_params()}")
    else:
        model.fit(train_df, feature_tup)
    
    predicted_views_per_day = model.predict(validation_df)
    val_score = model.get_score(validation_df)
    print_w_sep(f"Model Cross Validation Score (R^2): {val_score}")
    
    if args.s:
        now = datetime.now()
        file = MODEL_PATH/(f"val_score_{round(val_score*100)}" + data_name + "_" + model_dict["model_name"] + "_" + now.strftime("%m_%d_%Y__%H_%M"))
        model.save(file)
        print_w_sep("Saved file as:", file)
    