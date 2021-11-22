# imdb_scraper

Scrapes entries of movies off of imdb.<br>
Currently configured to scrape only movies that have ratings, were released in or after 1950, and are not adult films. Blacklists any ID values that are attributed to IMDb entries that aren't a movie or fail the outlined filters. Randomly queries ID values between a minimum and maximum, currently set to 1 -> 1160419 [Dune (2021)].<br>
Saves data in the following format:<br>
***[movieID, title, year, rating, vote count, genre, runtime, language]***

## Installation

Requires poetry to run.<br>
Install poetry via:<br>
***curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -***<br>
Then, in a terminal at imdb_scraper src:<br>
***poetry install***<br>
This installs the imdb_scraper package and all of it's dependencies.

## Running the imdb scraper

Run via poetry using:<br>
***poetry run scrape <n\>***<br>
Where ***n*** is the number of new movies to scrape

## Running the tmdb scraper

Run via poetry using:<br>
***poetry run scrape_tmdb <id_file.json\> <out_file.json\> <api_key\> <n\>***
Where ***id_file** is a file full of json objects with a field "id" defined that corresponds to a tmdb id,
***out_file*** is the file to read/write to/from, ***api_key*** is your api key for tmdb, and ***n*** is the number of non-adult movies to query in total.


## Merging imdb data files

Run via poetry using:<br>
***poetry run merge <from\> <to\> <is_blacklist\>***<br>
Where ***from*** is loaded along with ***to***, and the difference of ***from - to*** is appended to ***to***. ***is_blacklist*** specifies whether the file is a blacklist file, you can specify using True, true, T, t, or 1. Any other input evaluates to False.

## Dependencies 
1. lime (pip install lime)