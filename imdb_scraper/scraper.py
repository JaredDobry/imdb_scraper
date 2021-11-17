import argparse
from imdb import IMDb, Movie, IMDbError
import logging
import os
import random
import time


MAX_ID = 1160419  # Dune (2021)
MIN_ID = 1
FILE_PATH = "personal_scraped_data/movie_data.txt"
BLACKLIST_PATH = "personal_scraped_data/blacklist.txt"
SLEEP_TIME = 1  # Seconds

parser = argparse.ArgumentParser()
parser.add_argument("n", help="The number of movies to scrape from IMDb", type=int)
parser.add_argument("--progress_bar", type=bool, default=True, help="Flag to display progress bar")

def movie_to_str(movie: Movie) -> str:
    keys = movie.keys()
    movie_str = str(movie.movieID)
    movie_str += ";" + movie["title"]
    movie_str += ";" + str(movie["year"]) if "year" in keys else ";None"
    movie_str += ";" + str(movie["rating"]) if "rating" in keys else ";None"
    movie_str += ";" + str(movie["votes"]) if "votes" in keys else ";None"
    movie_str += (
        ";" + movie["genres"][0]
        if ("genres" in keys and movie["genres"][0])
        else ";None"
    )
    movie_str += (
        ";" + movie["runtimes"][0]
        if ("runtimes" in keys and movie["runtimes"][0])
        else ";None"
    )
    movie_str += (
        ";" + movie["languages"][0]
        if ("languages" in keys and movie["languages"][0])
        else ";None"
    )
    return movie_str


def str_to_movie(movie_str: str) -> list:
    out = []
    movie_split = movie_str.replace("\n", "").replace("\r", "").split(";")
    out.append(movie_split[0])
    out.append(movie_split[1])
    out.append(int(movie_split[2]) if movie_split[2] != "None" else None)
    out.append(float(movie_split[3]) if movie_split[3] != "None" else None)
    out.append(int(movie_split[4]) if movie_split[4] != "None" else None)
    out.append(movie_split[5] if movie_split[5] != "None" else None)
    out.append(int(movie_split[6]) if movie_split[6] != "None" else None)
    out.append(movie_split[7] if movie_split[7] != "None" else None)
    return out


def movie_list_to_str(movie_list: list) -> str:
    return f"{movie_list[0]};{movie_list[1]};{movie_list[2]};{movie_list[3]};{movie_list[4]};{movie_list[5]};{movie_list[6]};{movie_list[7]}"


def save_data(filepath: str, movies: list[str], append: bool = False) -> None:
    if len(movies) == 0:
        return
    try:
        first = False if append else True
        fw = open(filepath, "a") if append else open(filepath, "w")
        for movie in movies:
            if first:
                fw.write(movie)
                first = False
            else:
                fw.write("\n" + movie)
        fw.close()
    except IOError as e:
        logging.error(f"Error writing to file: {filepath} - {e}")


def load_data(filepath: str) -> list:
    if not os.path.exists(filepath):
        logging.info(f"File {filepath} doesn't exist, assuming first run...")
        return []
    try:
        fr = open(filepath, "r")
        lines = fr.readlines()
        fr.close()
        out = []
        for line in lines:
            if (
                len(line.replace("\n", "").replace("\r", "").replace(" ", "")) > 0
            ):  # Blank lines
                out.append(str_to_movie(line))
        return out
    except IOError as e:
        logging.error(f"Error reading from file: {filepath} - {e}")
        raise e


def save_blacklist(filepath: str, bl_ids: list[int], append: bool = False) -> None:
    if len(bl_ids) == 0:
        return
    try:
        first = False if append else True
        fw = open(filepath, "a") if append else open(filepath, "w")
        for bl_id in bl_ids:
            if first:
                fw.write(str(bl_id))
                first = False
            else:
                fw.write("\n" + str(bl_id))
        fw.close()
    except IOError as e:
        logging.error(f"Error writing to file: {filepath} - {e}")


def load_blacklist(filepath: str) -> list[int]:
    if not os.path.exists(filepath):
        logging.info(f"File {filepath} doesn't exist, assuming first run...")
        return []
    try:
        fr = open(filepath, "r")
        lines = fr.readlines()
        fr.close()
        out = []
        for line in lines:
            if len(line.replace("\n", "")) > 0:  # Blank lines
                out.append(int(line.replace("\n", "")))
        return out
    except IOError as e:
        logging.error(f"Error reading from file: {filepath} - {e}")
        raise e


def find_new_movie(movie_data: list, blacklist: list[int]) -> int:
    while True:
        new_id = random.randrange(MIN_ID, MAX_ID)
        if new_id not in blacklist and new_id not in [m[0] for m in movie_data]:
            return new_id


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Load existing data
    logging.info("Loading existing data")
    movie_data = load_data(FILE_PATH)
    blacklist = load_blacklist(BLACKLIST_PATH)
    logging.info("Finished loading data")
    write_cache = []
    blacklist_cache = []

    # Init
    args = parser.parse_args()
    n = args.n
    progress_bar_flag = args.progress_bar
    random.seed(None)
    ia = IMDb(loggingLevel=logging.ERROR)
    count = 0
    first_write = len(movie_data) == 0
    blacklist_first_write = len(blacklist) == 0

    try:
        while count < n:  # Main loop
            time.sleep(SLEEP_TIME)
            movie_id = find_new_movie(movie_data, blacklist)
            try:
                movie = ia.get_movie(movie_id)
            except IMDbError as e:
                logging.error(f"Error getting movie ID: {movie_id} - {e}")
                blacklist.append(movie_id)
                blacklist_cache.append(movie_id)
                continue

            if progress_bar_flag:
                os.system('cls' if os.name == 'nt' else 'clear')
                dec_complete = count/n
                progress = int(round(50 * dec_complete))
                logging.info("[" + progress * "|" + (50-progress) * " " + "]" + f" {dec_complete*100}%")

            # Do some filtering
            keys = movie.keys()
            if "kind" not in keys or movie["kind"] != "movie":
                logging.info(f"ID {movie_id} was not a movie")
                blacklist.append(movie_id)
                blacklist_cache.append(movie_id)
                continue
            if "rating" not in keys:
                logging.info(f"Movie {movie_id} has no ratings")
                blacklist.append(movie_id)
                blacklist_cache.append(movie_id)
                continue
            if "year" not in keys or movie["year"] < 1950:
                logging.info(f"Movie {movie_id} has no year or is too old (<1950)")
                blacklist.append(movie_id)
                blacklist_cache.append(movie_id)
                continue
            if "genres" in keys and "Adult" in movie["genres"]:
                logging.info(f"Movie {movie_id} was an adult film")
                blacklist.append(movie_id)
                blacklist_cache.append(movie_id)
                continue

            logging.info(f"Got Movie: {movie}")
            movie_str = movie_to_str(movie)
            movie_data.append(movie_str)
            write_cache.append(movie_str)
            count += 1
            if len(write_cache) >= 10:
                logging.info("Writing movie cache")
                save_data(FILE_PATH, write_cache, not first_write)
                first_write = False
                write_cache = []
                logging.info("Finished writing movie cache")
                logging.info(f"Count is {count}")
            if len(blacklist_cache) >= 10:
                logging.info("Writing blacklist cache")
                save_blacklist(
                    BLACKLIST_PATH, blacklist_cache, not blacklist_first_write
                )
                blacklist_first_write = False
                blacklist_cache = []
                logging.info("Finished writing blacklist cache")
    except KeyboardInterrupt:
        if progress_bar_flag:
            os.system('cls' if os.name == 'nt' else 'clear')
        logging.error("Keyboard Interrupt. Shutting down scraper.")
    except Exception as exc:
        if progress_bar_flag:
            os.system('cls' if os.name == 'nt' else 'clear')
        logging.error(f"Encountered exception while scraping IMDb: {exc}")
        raise exc
    finally:
        if progress_bar_flag:
            dec_complete = count/n
            progress = int(round(50 * dec_complete))
            logging.info("[" + progress * "|" + (50-progress) * " " + "]" + f" {dec_complete*100}%")
        logging.info("Doing final writes")
        save_data(FILE_PATH, write_cache, not first_write)  # Save data
        save_blacklist(BLACKLIST_PATH, blacklist_cache, not blacklist_first_write)
        logging.info("Final writes complete")


if __name__ == "__main__":
    logging.info("Starting up scraper")
    main()
