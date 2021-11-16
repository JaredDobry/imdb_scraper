import argparse
import logging
import os
from imdb_scraper.scraper import (
    load_blacklist,
    load_data,
    movie_list_to_str,
    save_blacklist,
    save_data,
)

parser = argparse.ArgumentParser()
parser.add_argument("fr", help="The file to pull extra information from", type=str)
parser.add_argument("to", help="The file to append the extra information to", type=str)
parser.add_argument(
    "is_blacklist",
    help="Set to [F,f](alse) if you are merging two movie lists, set to [T,t](rue) if you are merging two blacklists",
    type=str,
)


def merge():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    if not os.path.exists(args.fr):
        logging.error(f"File {args.fr} does not exist")
        raise FileNotFoundError
    if not os.path.exists(args.to):
        logging.error(f"File {args.to} does not exist")
        raise FileNotFoundError

    args.is_blacklist = (
        args.is_blacklist == "True"
        or args.is_blacklist == "true"
        or args.is_blacklist == "T"
        or args.is_blacklist == "t"
        or args.is_blacklist == "1"
        or args.is_blacklist == 1
    )

    # Load data in
    logging.info("Loading data for merge")
    fr = load_blacklist(args.fr) if args.is_blacklist else load_data(args.fr)
    to = load_blacklist(args.to) if args.is_blacklist else load_data(args.to)
    logging.info("Data loaded")

    # Perform merge
    logging.info("Merging lists")
    different = []
    if args.is_blacklist:
        for movie_id in fr:
            if movie_id not in to:
                different.append(movie_id)
    else:
        for item in fr:
            if item[0] not in [x[0] for x in to]:
                different.append(movie_list_to_str(item))
    logging.info(
        f"Merge complete. {len(different)} elements being added to {args.to} from {args.fr}"
    )

    # Save the output
    save_blacklist(args.to, different, True) if args.is_blacklist else save_data(
        args.to, different, True
    )
    logging.info(f"Append write to {args.to} complete.")


if __name__ == "__main__":
    merge()
