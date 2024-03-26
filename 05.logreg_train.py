import argparse
from lib.validate import is_csv
import pandas as pd
import numpy as np
from lib.print import success, info

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def batch_gradient_decent(x: pd.DataFrame, y: pd.Series):
    alpha = 0.5
    iterations = 1000
    thetas = np.zeros(x.shape[1])

    for _ in range(iterations):
        z = np.dot(x, thetas)
        h = sigmoid(z)
        gradient = np.dot(x.T, (h - y)) / y.size
        thetas -= alpha * gradient
    return thetas

def get_thetas(house: str, x: pd.DataFrame, y: pd.Series):
    # Change house to 1 or 0
    y = y.apply(lambda h: 1 if h == house else 0)
    # Calculate thetas
    thetas = batch_gradient_decent(x, y)
    return thetas

def predict(x: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    houses = weights.columns
    possibilities = np.zeros(len(houses))
    y = np.empty(x.shape[0], dtype=object)

    i = 0
    for sample in x.values:
        for idx, house in enumerate(houses):
            possibilities[idx] = sigmoid(np.dot(sample, weights[house]))
        y[i] = houses[np.argmax(possibilities)]
        i += 1
    return y

def normalize(df: pd.DataFrame):
    min_values = df.min()
    max_values = df.max()
    normalized_df = (df - min_values) / (max_values - min_values)
    df[:] = normalized_df.values
    return df

def accuracy(y1: np.ndarray, y2: np.ndarray) -> float:
    return (y1 == y2).sum() / len(y1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="logreg_train: train a logistic regression model")
    parser.add_argument("csv_file", type=is_csv, help="Path to the .csv file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file, index_col="Index")
    df.drop(columns=['Arithmancy', 'Defense Against the Dark Arts', 'Potions', 'Care of Magical Creatures'], inplace=True)
    
    # Remove rows with NaN
    df.dropna(inplace=True)

    # Select numeric features
    x = normalize(df.select_dtypes(include='number'))
    y = df['Hogwarts House']

    houses = y.unique()
    weights = pd.DataFrame(columns=houses)
    weights.index.name = 'Thetas'

    print(info('Start training...'))
    for house in houses:
        thetas = get_thetas(house, x, y)
        weights[house] = thetas

    # Save weights
    weights.to_csv('weights.csv')
    print(f"Weights saved into {success('./weights.csv')}")
    print(info('Training done!'))

    # Test with train data
    print(info("\nTesting with train data..."))
    y2 = predict(x, weights)

    # Accuracy
    print(f"Accuracy: {success('{:.2%}'.format(accuracy(y.values, y2)))}")