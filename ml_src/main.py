from typing import Dict, List
import utils
import metrics
import pathlib
import numpy as np
from utils import Feature
from model_runner import ModelRunner
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

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
    print(train_df.info())
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
        Feature(("budget",), metrics.get_numeric),
        Feature(("belongs_to_collection",), metrics.get_belongs_to_collection),
        Feature(("genres",), metrics.get_genres),
        Feature(("original_language",), metrics.get_original_language),
        Feature(("popularity",), metrics.get_numeric),
        Feature(("release_date",), metrics.get_release_year),
        Feature(("revenue",), metrics.get_numeric),
        Feature(("runtime",), metrics.get_numeric),
        Feature(("spoken_languages",), metrics.get_num_spoken_languages),
        Feature(("vote_average", "vote_count",), metrics.get_vote_popularity),
    )
    
    model_type = LinearRegression(normalize=True)
    model = ModelRunner(model_type, is_grid_search=True)
    model.fit(train_df, feature_tup)
    # model.predict(test_df)
    # model.explain_py(train_df, test_df, 27)
    
    # print("Views per day predictions: ", model.predict(test_df))
    # print("Training score: ", model.get_score(test_df))
    # model.save(MODEL_PATH/"Hello")
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
    
    # genres_movie = lambda genre_dict_ls: [genre_dict["name"] for genre_dict in genre_dict_ls]
    # genre_ls = np.array([genres_movie(movie) for movie in train_df.genres], dtype=object).reshape(-1,1)
    # print(OneHotEncoder().fit_transform(genre_ls))
    