import argparse
import logging
from json import loads
from os import mkdir
from os.path import exists
from typing import List, Dict, Tuple
from random import seed, randint
from sklearn import linear_model
from pathlib import Path
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("filepath", type=str)
parser.add_argument("split_percent", type=float)


def is_none(item):
    if item is None:
        return True
    elif type(item) == str:
        return item.lower() == "none"
    return False


def is_zero(item: int):
    return item == 0


KEY_TABLE = {
    "belongs_to_collection": bool,
    "budget": (int, is_zero),
    "genres": List,  # One hot
    "id": int,
    "imdb_id": int,
    "overview": str,
    "popularity": (float, is_none),
    "production_companies": List,  # One hot?
    "production_countries": bool,  # Made in USA or elsewhere
    "release_date": (int, is_none),
    "revenue": (int, is_zero),
    "runtime": (int, is_zero),
    "spoken_languages": bool,  # One hot?
    "title": str,
    "views_per_day": (float, is_none),
    "vote_average": (float, is_zero),
    "vote_count": (int, is_zero),
}


def load_data(filepath: str) -> List[Dict]:
    if not exists(filepath):
        raise FileNotFoundError

    logging.info(f"Loading data file: {filepath}")

    # Figure out how many lines there are so we can pre-allocate an array
    lines = 0
    with open(filepath, "r") as f:
        for _ in f:
            lines += 1
    arr = [{}] * lines
    logging.info(f"Initialized list of size {lines}")

    # Perform load
    x = 0
    with open(filepath, "r") as f:
        for line in f:
            arr[x] = loads(line)
            x += 1

    logging.info("Load complete")
    return arr


def clean_data(data: List[Dict], keys: List[str], target: str) -> List[List]:
    logging.info("Cleaning data using keys: [")
    for key in keys:
        logging.info(key)
    logging.info(target)
    logging.info("]")

    cleaned = []
    for item in data:
        valid = True
        for key in item.keys():
            if key == target and KEY_TABLE[key][1](KEY_TABLE[key][0](item[key])):
                valid = False
                break
            elif key in keys and KEY_TABLE[key][1](KEY_TABLE[key][0](item[key])):
                valid = False
                break
        if valid:
            pared = []
            for key in keys:
                pared.append(KEY_TABLE[key][0](item[key]))
            pared.append(KEY_TABLE[target][0](item[target]))
            cleaned.append(pared)
    logging.info(f"{len(cleaned)} valid data entries being used")
    return cleaned


def partition_data(data: List[List], split: float) -> Tuple[List[List], List[List]]:
    assert split > 0
    assert split < 1

    # Need to pre-allocate when working with large datasets
    test_len = int(split * len(data))
    train_data = [[]] * (len(data) - int(split * len(data)))
    test_data = [[]] * test_len
    logging.info(
        f"Pre-allocated training array of size: {len(train_data)} and testing array of size: {len(test_data)}"
    )

    train_x = 0
    test_x = 0
    for item in data:
        if test_x == len(test_data):
            train_data[train_x] = item
            train_x += 1
        elif train_x == len(train_data):
            test_data[test_x] = item
            test_x += 1
        else:
            roll = randint(1, len(data)) <= test_len
            if roll:
                test_data[test_x] = item
                test_x += 1
            else:
                train_data[train_x] = item
                train_x += 1

    logging.info(
        f"Data partitioned into {len(train_data)} training entries and {len(test_data)} testing entries"
    )
    return train_data, test_data


def data_to_x_y(data: List[List]) -> Tuple[List[List], List[int]]:
    x = []
    y = []
    for item in data:
        x.append(item[:-1])
        y.append(item[-1])

    return x, y


def train_model(x: List[List], y: List[int], model_class, model_kwargs: dict = None):
    logging.info(f"Training model: {model_class.__name__}")
    if model_kwargs:
        clf = model_class(**model_kwargs)
    else:
        clf = model_class()
    clf.fit(x, y)
    logging.info("Training complete")
    return clf


def score_model(x: List[List], y: List[int], model_object):
    return model_object.score(x, y)


def write_file(data: List[List], filepath: Path):
    with open(filepath, "w") as f:
        first = True
        for item in data:
            if first:
                first = False
            else:
                f.write("\n")
            f.write(str(item))


def main():
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    rand_seed = 10
    seed(rand_seed)

    # Load data
    data = load_data(args.filepath)

    # Clean data
    keys = ["budget", "revenue", "vote_average", "vote_count"]
    target = "popularity"
    cleaned = clean_data(data, keys, target)

    # Partition data
    train_data, test_data = partition_data(cleaned, args.split_percent)
    train_x, train_y = data_to_x_y(train_data)
    test_x, test_y = data_to_x_y(test_data)

    # Create folder for model output
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    folder_path = Path(__file__).parent.parent.joinpath(f"models/{date_str}")
    mkdir(folder_path)

    models = [
        (linear_model.LinearRegression, None),
        (linear_model.Ridge, {"alpha": 0.5}),
        (linear_model.Lasso, {"max_iter": 10000, "alpha": 0.1}),
        (linear_model.ElasticNet, {"max_iter": 10000, "alpha": 1.0, "l1_ratio": 0.5}),
        (linear_model.Lars, None),
    ]

    for model_class, model_args in models:
        # Train model
        model = train_model(
            x=train_x, y=train_y, model_class=model_class, model_kwargs=model_args
        )

        # Score model
        score = score_model(test_x, test_y, model)

        logging.info(f"Model {model_class.__name__} got score: {score}")

        # Version
        model_folder_path = folder_path.joinpath(f"{model_class.__name__}")
        mkdir(model_folder_path)

        write_file(train_data, model_folder_path.joinpath("training_data.txt"))
        write_file(test_data, model_folder_path.joinpath("testing_data.txt"))
        with open(model_folder_path.joinpath("results.txt"), "w") as f:
            f.write(f"{score}")
        with open(model_folder_path.joinpath("keys.txt"), "w") as f:
            f.write(f"{keys}")
        with open(model_folder_path.joinpath("seed.txt"), "w") as f:
            f.write(f"{rand_seed}")


if __name__ == "__main__":
    main()
