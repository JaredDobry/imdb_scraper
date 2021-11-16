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

## Running the scraper

Run via poetry using:<br>
***poetry run scrape <n\>***<br>
Where n is the number of new movies to scrape
