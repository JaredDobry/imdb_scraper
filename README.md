# imdb_scraper

Scrapes entries of movies off of imdb.`<br>`
Currently configured to scrape only movies that have ratings, were released in or after 1950, and are not adult films. Blacklists any ID values that are attributed to IMDb entries that aren't a movie or fail the outlined filters. Randomly queries ID values between a minimum and maximum, currently set to 1 -> 1160419 [Dune (2021)].`<br>`
Saves data in the following format:`<br>`
***[movieID, title, year, rating, vote count, genre, runtime, language]***

## Installation

Requires poetry to run.`<br>`
Install poetry via:`<br>`
***curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -***`<br>`
Then, in a terminal at imdb_scraper src:`<br>`
***poetry install***`<br>`
This installs the imdb_scraper package and all of it's dependencies.

## Running the imdb scraper

Run via poetry using:`<br>`
***poetry run scrape <n\>***`<br>`
Where ***n*** is the number of new movies to scrape

## Running the tmdb scraper

Run via poetry using:`<br>`
***poetry run scrape_tmdb <id_file.json\> <out_file.json\> <api_key\> <n\>***
Where ***id_file** is a file full of json objects with a field "id" defined that corresponds to a tmdb id,
***out_file*** is the file to read/write to/from, ***api_key*** is your api key for tmdb, and ***n*** is the number of non-adult movies to query in total.

## Merging imdb data files

Run via poetry using:`<br>`
***poetry run merge <from\> <to\> <is_blacklist\>***`<br>`
Where ***from*** is loaded along with ***to***, and the difference of ***from - to*** is appended to ***to***. ***is_blacklist*** specifies whether the file is a blacklist file, you can specify using True, true, T, t, or 1. Any other input evaluates to False.

## Dependencies

1. lime (pip install lime)
2. numpy (pip install numpy)
3. pandas (pip install pandas)

## Running the Training Model

There are 2 methods to train the model. Each method allows the user to select a training model type (Linear Regression, Ridge Regression, Decision Tree Regression, and Random Forest Regression), whether they want to run a grid search of the model, and whether they want to save the model as a pickle file. We suggest one to use the Random Forest Regression as it has given us the best results. The options are:

1. Find the Jupyter notebook file "main.ipynb" and run each block in order. Some blocks are not necessary and for others, you may need to choose only one out of a few. The markdown should provide a decent understanding of a block's role so merely follow the instructions in markdown.
   This method provides a significant amount of customizability but has significant overhead and requires much work from whomever is running the code.
2. The ml_src/train.py script performs the same task as above. One must select a training model as a command line argument (i.e. "python train.py 'Random Forest'") and they can run a grid search with the optional parameter "-g" and/or save the model with the optional parameter "-s." Note, to see what models can be chosen, type "python train.py -h." This will provide a list of options one can use to save the model. Also note that each algorithm has a full name (e.g. Random Forest) and a shortcut name (e.g. RF). These correspond to the same algorithm and are just there for convenience.
   This method has less overhead and is our preferred method.

## Running the Model on the Testing Data

To run the model on the testing data, go to the jupyter script. The bottom of the jupyter script contains the code to load in a saved model (if desired, otherwise one must train a model first) and run the saved model with the testing data. The last box in the jupyter script displays the Lime explainer.
