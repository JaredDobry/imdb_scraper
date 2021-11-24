import argparse
from typing import Dict, List, Tuple, Union
import ml_src.utils as utils
import ml_src.metrics as metrics
import pathlib
import numpy as np
from tmdb_scraper.scraper import load_json
from ml_src.utils import Feature, unpickle_file
from ml_src.model_runner import ModelRunner
from sklearn.linear_model import LinearRegression
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# import seaborn as sns
import pandas as pd

# import matplotlib.pyplot as plt

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH / "StaticDB"
MODEL_PATH = PARENT_PATH / "models"

parser = argparse.ArgumentParser()
parser.add_argument("training", type=str)
parser.add_argument("testing", type=str)


def thin_dataframe(
    dataframe: pd.DataFrame, keys: List[str], in_place: bool = True
) -> Union[None, pd.DataFrame]:
    discard = []
    for key in dataframe.columns:
        if key not in keys:
            discard.append(key)
    out = dataframe.drop(columns=discard, inplace=in_place)
    return out


def drop_rows_on_cond(dataframe: pd.DataFrame, key: str) -> pd.DataFrame:
    if KEY_TYPE_CONVERSION[key][1] == "is_zero":
        for item in dataframe:
            if item[key] == 0:
                dataframe.drop(index=item.index, inplace=True)

    return dataframe


def is_zero(val) -> bool:
    if val == 0:
        return True
    if type(val) == str and val == "0":
        return True
    return False


def is_none(val) -> bool:
    if val is None:
        return True
    if type(val) == str and val.lower() == "none":
        return True
    return False


def clean_data(
    data: List[Dict], keys: List[str], target_key: str
) -> Tuple[List[List], List]:
    x = []
    y = []
    for item in data:
        drop_me = False
        features = []
        target_val = None
        for key in item.keys():
            if key == target_key:  # Must have a valid target key value
                target_val = KEY_TYPE_CONVERSION[key][0](item[key])
                if KEY_TYPE_CONVERSION[key][1](target_val):
                    drop_me = True
                    break
            elif key in keys:  # Do we care to process
                converted = KEY_TYPE_CONVERSION[key][0](item[key])
                if KEY_TYPE_CONVERSION[key][1](converted):
                    drop_me = True
                    break
                else:
                    features.append(converted)
        if drop_me or target_val is None:
            continue
        else:
            x.append(features)
            y.append(target_val)
    return x, y


# key: (conversion function, drop function)
KEY_TYPE_CONVERSION = {
    "belongs_to_collection": bool,
    "budget": (int, is_zero),
    "genres": List,  # One hot
    "id": int,
    "imdb_id": int,
    "overview": str,
    "popularity": (float, is_none),
    "production_companies": List,  # One hot?
    "production_countries": bool,  # Made in USA or elsewhere
    "release_date": int,
    "revenue": (int, is_zero),
    "runtime": int,
    "spoken_languages": bool,  # One hot?
    "title": str,
    "views_per_day": (float, is_none),
    "vote_average": (float, is_none),
    "vote_count": (int, is_zero),
}


def model_pipeline(
    training_filepath: str, testing_filepath: str, keys: List[str], target_key: str
):
    # Load data
    if pathlib.Path(training_filepath).suffix == ".pickle":
        training_data = unpickle_file(pathlib.Path(training_filepath))
    elif pathlib.Path(training_filepath).suffix == ".json":
        training_data = load_json(training_filepath)
    else:
        raise TypeError("Only .pickle and .json files supported")

    if pathlib.Path(testing_filepath).suffix == ".pickle":
        testing_data = unpickle_file(pathlib.Path(testing_filepath))
    elif pathlib.Path(testing_filepath).suffix == ".json":
        testing_data = load_json(testing_filepath)
    else:
        raise TypeError("Only .pickle and .json files supported")
    logging.info("Files loaded successfully")

    # Data Cleaning
    logging.info(f"Cleaning data")

    x, y = clean_data(training_data, keys, target_key)

    logging.info("Running classifier")
    clf = LinearRegression()
    clf.fit(x, y)

    # Scoring
    x, y = clean_data(testing_data, keys, target_key)

    logging.info("Scoring classifier")
    score = clf.score(x, y)
    logging.info(f"Classifier got score: {score}")


def train_model():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    keys = ["budget", "revenue"]
    model_pipeline(args.training, args.testing, keys, "views_per_day")


if __name__ == "__main__":
    train_df = utils.unpickle_df(DB_PATH / "train_movies.pickle")
    test_df = utils.unpickle_df(DB_PATH / "test_movies.pickle")

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
        Feature(("budget",), metrics.get_numeric),
        Feature(("belongs_to_collection",), metrics.get_belongs_to_collection),
        Feature(("genres",), metrics.get_genres),
        Feature(("original_language",), metrics.get_original_language),
        Feature(("popularity",), metrics.get_numeric),
        Feature(("release_date",), metrics.get_release_year),
        Feature(("revenue",), metrics.get_numeric),
        Feature(("runtime",), metrics.get_numeric),
        Feature(("spoken_languages",), metrics.get_num_spoken_languages),
        Feature(
            (
                "vote_average",
                "vote_count",
            ),
            metrics.get_vote_popularity,
        ),
    )

    model_type = LinearRegression(normalize=True)
    model = ModelRunner(model_type, is_grid_search=True)
    model.fit(train_df, feature_tup)
    # model.predict(test_df)
    # model.explain_py(train_df, test_df, 27)

    print("Views per day predictions: ", model.predict(test_df))
    print("Training score: ", model.get_score(test_df))
    model.save(MODEL_PATH / "Hello")
    print("Best score:", model.get_best_score())
    print("Best params:", model.get_best_params())

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
