# Model Evaluation: Finally, using the trained model, we will evaluate its 
# performance and make predictions. At the end of this script, we will have a 
# new dataset in data containing the predictions, along with a scores.json file 
# in the metrics directory that will capture evaluation metrics of our model 
# (e.g., MSE, R2).


import logging
from pathlib import Path
import pickle
import json

import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score


def evaluate():
    projDir = Path(__file__).absolute().parent.parent.parent

    processedDir = projDir / "data/processed"
    X_testFile = processedDir / "scaled" / "X_test_scaled.csv"
    y_testFile = processedDir / "split"/ "y_test.csv"
    logging.debug(f"Input: {X_testFile=}")
    logging.debug(f"Input: {y_testFile=}")

    X_test = pd.read_csv(X_testFile)
    y_test = pd.read_csv(y_testFile)
    y_test = y_test.squeeze()

    modelFile = projDir / "models/models" / "trained_model.pkl"
    logging.debug(f"Input: {modelFile=}")
    with open(modelFile, 'rb') as f:
        model = pickle.load(f)

    y_predict = model.predict(X_test)
    y_predict = pd.Series(y_predict, name=y_test.name)

    scores = {
        'RMSE': root_mean_squared_error(y_test, y_predict),
        'R2': r2_score(y_test, y_predict),
    }
    logging.info(f"{scores=}")

    predictFile = projDir / "data/predicted" / "prediction.csv"
    predictFile.parent.mkdir(parents=True, exist_ok=True)
    y_predict.to_csv(predictFile, index=False)

    scoresFile = projDir / "metrics" / "scores.json"
    scoresFile.parent.mkdir(parents=True, exist_ok=True)
    with open(scoresFile, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
    )

    evaluate()
