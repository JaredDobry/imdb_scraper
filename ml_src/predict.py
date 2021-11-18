import utils
import metrics
import pathlib
from model_runner import ModelRunner, load_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
# import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH/"StaticDB"
MODEL_PATH = PARENT_PATH/"models"

if __name__ == "__main__":
    test_df = utils.unpickle_df(DB_PATH/"test_movies.pickle")  
    model = load_model(MODEL_PATH/"LinReg_budget_isCollection")
    
    print("Views per day predictions: ", model.predict(test_df))
    print("Training score: ", model.score(test_df))
