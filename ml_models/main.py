from os import name
import utils
import metrics
import pathlib
from model_base import ModelRunner
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

if __name__ == "__main__":
    train_df = utils.unpickle_df(DB_PATH/"train_movies.pickle")
    test_df = utils.unpickle_df(DB_PATH/"test_movies.pickle")  

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
            
    model_type = LinearRegression(normalize=True)
    model = ModelRunner(model_type)
    model.fit(train_df, feature_tup)
    print("Views per day predictions: ", model.predict(test_df))
    print("Training score: ", model.score(test_df))

    # print(50*"=")

    # model_type = Ridge(0.4, normalize=True)
    # model2 = ModelRunner(model_type)
    # model2.fit(train_df, feature_tup)
    # print("Views per day predictions: ", model2.predict(test_df))
    # print("Training score: ", model2.score(test_df))
    
    
  
    # x = 4
    # import pickle
    # with open("x_val.pickle", "ab") as f:
    # #     pickle.dump(x, f)
    #     print(pickle.load(f)

    
    # The line below will give you info about every column
    # print(train_df.info())
    # print(30*"=",'\n')
    
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