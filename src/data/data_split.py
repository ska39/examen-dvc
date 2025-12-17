# Data Splitting: Split the data into training and testing sets.
# Our target variable is silica_concentrate, located in the last column of 
# the dataset. This script will produce 4 datasets 
# (X_test, X_train, y_test, y_train) that you can store in data/processed.

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def data_split():
    projDir = Path(__file__).absolute().parent.parent.parent

    rawFile = projDir / "data/raw" / "raw.csv"
    rawDf = pd.read_csv(rawFile)

    y = rawDf.pop('silica_concentrate')
    X = rawDf.drop(columns=['date'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    outDir = projDir / "data/processed"
    X_train.to_csv(outDir / "X_train.csv")
    X_test.to_csv(outDir / "X_test.csv")
    y_train.to_csv(outDir / "y_train.csv")
    y_test.to_csv(outDir / "y_test.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s: %(message)s',
    )

    data_split()
