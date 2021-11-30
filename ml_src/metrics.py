import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List, Dict, Tuple

Feature = namedtuple("Feature", ["feature_keys", "handle"])
Category = namedtuple("Category", ["category_vals", "category_column_names"])


def get_numeric(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category([[float(x) for x in df[names[0]]]], names)


def get_belongs_to_collection(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category([[1 if d else 0 for d in df[names[0]]]], ("belongs_to_collection",))


def get_genres(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    genres = list(
        np.unique(
            [
                genre_dict["name"]
                for movie_genre in df[names[0]]
                for genre_dict in movie_genre
            ]
        )
    )

    def get_row(genre_dict_ls: List[Dict[str, str]]) -> List[int]:
        row = [0] * len(genres)
        for genre_dict in genre_dict_ls:
            index = genres.index(genre_dict["name"])
            row[index] = 1
        return row

    return Category(
        np.array(
            [get_row(movie_genre_ls) for movie_genre_ls in df[names[0]]]
        ).T.tolist(),
        tuple(genres),
    )


def get_original_language(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category(
        [list(map(int, (df["original_language"] == "en").to_list()))], ("lang_orig_is_English",)
    )


def get_release_year(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category(
        [
            [
                1949 if date == "null" else int(date.partition("-")[0])
                for date in df[names[0]]
            ]
        ],
        ("release_year",),
    )


def get_num_spoken_languages(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 1
    return Category([[len(d_ls) for d_ls in df[names[0]]]], ("num_spoken_langs",))


def get_vote_popularity(df: pd.DataFrame, names: Tuple[str]) -> Category:
    assert len(names) == 2
    return Category(
        [[float(count) * float(avg) for count, avg in zip(df[names[0]], df[names[1]])]],
        ("vote_popularity",),
    )


# Features of Interest:
# Production Companies I think is a lowkey important feature, maybe from country to country
# it varies, but the well-watched movies are made by the same big name production companies.
