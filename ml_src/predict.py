import utils
import pathlib
from model_runner import load_model

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH / "StaticDB"
MODEL_PATH = PARENT_PATH / "models"

if __name__ == "__main__":
    """
    This script is for loading in pickled datasets and running their models on the testing data.
    """   
    
    test_df = utils.unpickle_df(DB_PATH / "test_movies.pickle")
    model = load_model(MODEL_PATH / "LinReg_budget_isCollection")

    print("Views per day predictions: ", model.predict(test_df))
    print("Training score: ", model.score(test_df))
