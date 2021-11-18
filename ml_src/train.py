import utils
import metrics
import pathlib
from model_runner import ModelRunner
from sklearn.linear_model import LinearRegression

PARENT_PATH = pathlib.Path("__dir__").parent.resolve()
DB_PATH = PARENT_PATH/"StaticDB"
MODEL_PATH = PARENT_PATH/"models"

if __name__ == "__main__":
    train_df = utils.unpickle_df(DB_PATH/"train_movies.pickle")
        
    feature_tup = (
        utils.Feature(("budget",), metrics.get_budget),
        utils.Feature(("belongs_to_collection",), metrics.get_belongs_to_collection),
    )
            
    model_type = LinearRegression(normalize=True)
    model = ModelRunner(model_type)
    model.fit(train_df, feature_tup)
    model.save(MODEL_PATH/"LinReg_budget_isCollection")
    