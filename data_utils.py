import pandas as pd

def load_data():

    test_data_path = "data/test.csv"
    train_data_path = "data/train.csv"
    test_data = pd.read_csv(test_data_path)
    train_data = pd.read_csv(train_data_path)
    train_Y = train_data.loc[:, "target"]
    train_X = train_data.iloc[:, 2:]
    test_X = test_data[:, 1]

    return train_X, train_Y, test_X





