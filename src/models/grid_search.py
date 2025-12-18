# GridSearch for Best Parameters: Decide on the regression model to implement 
# and the parameters to test. At the end of this script, we will have the best 
# parameters saved as a .pkl file in the models directory.

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def grid_search():
    projDir = Path(__file__).absolute().parent.parent.parent

    processedDir = projDir / "data/processed"
    X_trainFile = processedDir / "scaled" / "X_train_scaled.csv"
    y_trainFile = processedDir / "split" / "y_train.csv"
    logging.debug(f"Input: {X_trainFile=}")
    logging.debug(f"Input: {y_trainFile=}")

    X_train = pd.read_csv(X_trainFile)
    y_train = pd.read_csv(y_trainFile)
    y_train = y_train.squeeze()

    model = HistGradientBoostingRegressor(random_state=42)

    paramGrid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [None, 10],
        'min_samples_leaf': [20, 100],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=paramGrid,
        scoring='neg_mean_squared_error',
        cv=5
    )

    search.fit(X_train, y_train)
    bestParams = search.best_params_
    bestRMSE = np.sqrt(-1*search.best_score_)

    logging.info(f"Found: {bestParams=}")
    logging.info(f"Training score: {bestRMSE=}")

    paramsFile = projDir / "models/data" / "best_params.pkl"
    logging.debug(f"Saving best parameters: {paramsFile=}")
    paramsFile.parent.mkdir(parents=True, exist_ok=True)
    with open(paramsFile, 'wb') as f:
        pickle.dump(bestParams, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
    )

    grid_search()
