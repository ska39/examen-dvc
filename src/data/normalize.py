# Data Normalization: As you may notice, the data varies widely in scale, 
# so normalization is necessary. You can use existing functions to construct 
# this script. As output, this script will create two new datasets 
# (X_train_scaled, X_test_scaled) which you will also save in data/processed.

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize():
    projDir = Path(__file__).absolute().parent.parent.parent

    splitDir = projDir / "data/processed/split"
    trainDataFile = splitDir / "X_train.csv"
    testDataFile = splitDir / "X_test.csv"
    logging.debug(f"Input: {trainDataFile=}")
    logging.debug(f"Input: {testDataFile=}")

    X_train = pd.read_csv(trainDataFile)
    X_test = pd.read_csv(testDataFile)

    if not X_train.columns.equals(X_test.columns):
        raise Exception('Train and test data columns do not match!')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    scaledDir = projDir / "data/processed/scaled"
    scaledDir.mkdir(parents=True, exist_ok=True)
    trainDataScaledFile = scaledDir / "X_train_scaled.csv"
    testDataScaledFile = scaledDir / "X_test_scaled.csv"
    logging.debug(f"Output: {trainDataScaledFile=}")
    logging.debug(f"Output: {testDataScaledFile=}")

    X_train_scaled.to_csv(trainDataScaledFile, index=False)
    X_test_scaled.to_csv(testDataScaledFile, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s',
    )

    normalize()
