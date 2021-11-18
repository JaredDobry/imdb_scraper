import argparse
import json
import tmdbsimple as tmdb
import os
import logging
import signal
from multiprocessing import Process, Queue, Event
from queue import Empty


NUM_WORKERS = 4


parser = argparse.ArgumentParser()
parser.add_argument(
    "json_path", help="The path to the TMDb json archive to pull movies from", type=str
)
parser.add_argument(
    "out_file", help="The filename/path to file you want to write to", type=str
)
parser.add_argument("api_key", help="Your API key", type=str)
parser.add_argument(
    "--progress_bar", type=bool, default=True, help="Flag to display progress bar"
)
args = parser.parse_args()
write_cache = []


def write_out(filepath: str) -> None:
    try:
        is_append = os.path.exists(filepath)
        fw = open(filepath, "a") if is_append else open(filepath, "w")
        logging.info(f"{'Appending' if is_append else 'Writing'} to file {filepath}.")
        first = not is_append
        for movie_dict in write_cache:
            if first:
                json.dump(movie_dict, fw)
                first = False
            else:
                fw.write("\n")
                json.dump(movie_dict, fw)
        fw.close()
        logging.info(f"{'Append' if is_append else 'Write'} complete.")
    except IOError as e:
        logging.error(e)


def kill_handler(_, __):
    logging.error("kill disabled while writing")


def get_movie(movie_id: int) -> dict:
    movie = tmdb.Movies(movie_id)
    response = movie.info()
    return response


def worker(work_queue: Queue, output_queue: Queue, kill: Event, idle: Event):
    while not kill.is_set():
        try:
            movie_id = work_queue.get_nowait()
            idle.clear()
            movie = get_movie(movie_id)
            output_queue.put(movie)
        except Empty:
            idle.set()
            continue


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("TMDb scraper setup")
    tmdb.API_KEY = args.api_key

    # Load the json dump
    logging.info("Loading json dump")
    to_get = []
    fr = open(args.json_path, "r")
    while line := fr.readline():
        to_get.append(json.loads(line)["id"])
    fr.close()
    logging.info("Json dump loaded")

    # Open the output file if it exists to check where we are progress wise
    if os.path.exists(args.out_file) and os.path.isfile(args.out_file):
        logging.info(f"Reading {args.out_file}")
        processed_ids = []
        fr = open(args.out_file, "r")
        while line := fr.readline():
            processed_ids.append(json.loads(line)["id"])
        fr.close()

        # Diff the progress we already made from to_get
        for movie_id in to_get:
            if movie_id in processed_ids:
                to_get.remove(movie_id)
        logging.info("Done processing existing data")

    # Process setup
    logging.info("Spawning sub-processes")
    kill_event = Event()
    work_queue = Queue()
    output_queue = Queue()

    # Create work tasks
    num_tasks = len(to_get)
    logging.info(f"Inserting {num_tasks} work items into queue")
    for movie_id in to_get:
        work_queue.put(movie_id)

    # Spawn processes
    processes = []
    for i in range(NUM_WORKERS):
        idle_event = Event()
        processes.append(
            [
                Process(
                    target=worker,
                    args=(work_queue, output_queue, kill_event, idle_event),
                    daemon=True,
                ),
                idle_event,
            ]
        )

    # Start
    for p in processes:
        p[0].start()
    logging.info("Sub-processes started")

    # Process output
    count = 0
    all_idle = False
    try:
        while not all_idle:
            # See if they are idle
            all_idle = True
            for p in processes:
                if not p[1].is_set():
                    all_idle = False
                    break

            # Try to get stuff to output
            while True:
                try:
                    movie = output_queue.get_nowait()
                    write_cache.append(movie)
                    count += 1

                    if args.progress_bar:
                        os.system("cls" if os.name == "nt" else "clear")
                        progress = count / num_tasks
                        bars = int(round(50 * progress))
                        progress_str = (
                            "["
                            + bars * "|"
                            + (50 - bars) * " "
                            + "] {percent:.2f}% - Last movie: {title}"
                        )
                        logging.info(
                            progress_str.format(
                                percent=progress * 100, title=movie["title"]
                            )
                        )
                    else:
                        logging.info(f"Got movie: {movie['title']}")
                except Empty:
                    break
    except KeyboardInterrupt:  # If interrupted during processing
        logging.error("Received KeyboardInterrupt, gracefully shutting down...")
        signal.signal(signal.SIGINT, kill_handler)
        kill_event.set()
        for p in processes:
            p[0].join()
        # Get whatever remains in the output queue
        while True:
            try:
                movie = output_queue.get_nowait()
                write_cache.append(movie)
                logging.info(f"Got movie: {movie['title']}")
            except Empty:
                break
        write_out(args.out_file)
        return

    # Join
    logging.info("Joining sub-processes")
    kill_event.set()
    for p in processes:
        p[0].join()

    # Write out the new data
    write_out(args.out_file)

    logging.info("TMDb scrape complete")


if __name__ == "__main__":
    main()
