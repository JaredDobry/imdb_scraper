from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from collections import namedtuple

# TODO: @Jared and Hyun, every one of these functions should return a
# List[List[numeric]] -> Since we need one hot encoding, many of these will
# return a two dimensional data collection. Therefore, the get_training_nparray
# function just assumes that it data will always be 2-dimensional

Category = namedtuple("Category", ["category_vals", "category_column_names"])

def __get_numeric__(ds: pd.Series) -> List[List[float]]:
    return [[float(x) for x in ds]]

def get_belongs_to_collection(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category([[1 if d else 0 for d in df[names[0]]]], names)

def get_budget(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category(__get_numeric__(df[names[0]]), names)

def get_genres(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    genres = list(np.unique([genre_dict["name"] for movie_genre in df[names[0]] for genre_dict in movie_genre]))
    
    def get_row(genre_dict_ls: List[Dict[str, str]]) -> List[int]:
        row = [0] * len(genres)
        for genre_dict in genre_dict_ls:
            index = genres.index(genre_dict["name"])
            row[index] = 1
        return row
    
    return Category(np.array([get_row(movie_genre_ls) for movie_genre_ls in df[names[0]]]).T.tolist(), tuple(genres))
    
# Features of Interest:
# Popularity is definitely important, seems to be an aggregate of vote count, # of favorited
# and # of times added to watchlist
# Revenue to me is more telling than budget, while there are some terrible movies that
# rake in money, usually the well-watched ones have grossed a lot of revenue.
# Vote Count is pretty important, because regardless of good or bad votes, a high vote count
# suggests a lot of views.
# Production Companies I think is a lowkey important feature, maybe from country to country
# it varies, but the well-watched movies are made by the same big name production companies.
    

