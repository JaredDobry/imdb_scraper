import utils
import metrics
import pathlib
from model_runner import ModelRunner
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH/"StaticDB"
MODEL_PATH = PARENT_PATH/"models"

if __name__ == "__main__":
    train_df = utils.unpickle_df(DB_PATH/"train_movies.pickle")
    test_df = utils.unpickle_df(DB_PATH/"test_movies.pickle")  

    # The lines below will give you info about every column
    # print(train_df.info())
    # print(test_df.info())

    drop_ls = [
        "id",
        "tmdb_id",
        "imdb_id",
        "title",
        "original_title",
        "adult",
        "homepage",
        "overview",
        "poster_path",
        "production_companies",
        "production_countries",
        "status",
    ]
        
    feature_tup = (
        utils.Feature(("budget",), metrics.get_budget),
        utils.Feature(("belongs_to_collection",), metrics.get_belongs_to_collection),
    )

    # params = {'alpha': np.logspace(start=-9, stop=9, num=500), 'normalize': [True, False]}
    # model_type = GridSearchCV(Ridge(), params, cv=10)
    
    model_type = LinearRegression(normalize=True)
    model = ModelRunner(model_type, is_grid_search=True)
    model.fit(train_df, feature_tup)
    
    print("Views per day predictions: ", model.predict(test_df))
    print("Training score: ", model.score(test_df))
    # print("Best score:", model.get_best_score())
    # print("Best params:", model.get_best_params())

    # print(50*"=")

    # model_type = Ridge(0.4, normalize=True)
    # model2 = ModelRunner(model_type)
    # model2.fit(train_df, feature_tup)
    # print("Views per day predictions: ", model2.predict(test_df))
    # print("Training score: ", model2.score(test_df))


    # This line will print all the "genres" column
    # print(train_df["genres"])
    
    # This line will show you all of the different possibilities for the spoken language category
    # print(train_df["spoken_languages"].value_counts())
    


    # IGNORE. Gives a heat map of how features correlate together.
    # genres = pd.get_dummies(train_df["genres"], drop_first=True)
    # train_df = pd.concat([train_df, genres], axis=1)
    # print(train_df)
    # ax = sns.set_context('paper')
    # ax = plt.figure(figsize=(7,7))
    # feature_df = train_df[feature_ls]
    # corr = feature_df.corr()
    # ax = sns.heatmap(corr, annot=True)
    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)
    # plt.show()