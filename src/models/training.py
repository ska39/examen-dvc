# Model Training: Using the parameters found through GridSearch, we will train 
# the model and save the trained model in the models directory.

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def training():
    projDir = Path(__file__).absolute().parent.parent.parent
    
    processedDir = projDir / "data/processed"
    X_trainFile = processedDir / "X_train_scaled.csv"
    y_trainFile = processedDir / "y_train.csv"
    logging.debug(f"Input: {X_trainFile=}")
    logging.debug(f"Input: {y_trainFile=}")

    X_train = pd.read_csv(X_trainFile)
    y_train = pd.read_csv(y_trainFile)
    y_train = y_train.squeeze()
    
    paramsFile = projDir / "models/data" / "best_params.pkl"
    logging.debug(f"Input: {paramsFile=}")

    with open(paramsFile, 'rb') as f:
        bestParams = pickle.load(f)
    
    model = HistGradientBoostingRegressor(random_state=42, **bestParams)    
    model.fit(X_train, y_train)

    modelFile = projDir / "models/models" / "trained_model.pkl"
    logging.debug(f"Saving trained model: {modelFile=}")
    modelFile.parent.mkdir(parents=True, exist_ok=True)
    with open(modelFile, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s: %(message)s',
    )

    training()
