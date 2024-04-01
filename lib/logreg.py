import numpy as np
import pandas as pd
from lib.math import math_mean, math_std

LEARNING_RATE = 0.5
ITERATIONS = 1000

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def batch_gradient_decent(x: pd.DataFrame, y: pd.Series):
    thetas = np.zeros(x.shape[1])
    for _ in range(ITERATIONS):
        # (1600, 9) * (9 x 1)
        z = np.dot(x, thetas)
        h = sigmoid(z)
        gradient = np.dot(x.T, (h - y)) / y.size
        thetas -= LEARNING_RATE * gradient
    return thetas

def mini_batch_gradient_decent(x: pd.DataFrame, y: pd.Series):
    thetas = np.zeros(x.shape[1])
    batch_size = 5
    m = len(x)
    indices = np.random.permutation(m)
    x_shuffled = x.iloc[indices]
    y_shuffled = y.iloc[indices]
    start_idx = 0

    for _ in range(ITERATIONS):
        start_idx = start_idx % m
        end_idx = (start_idx + batch_size - 1) % m

        x_batch = x_shuffled.iloc[start_idx:end_idx]
        y_batch = y_shuffled.iloc[start_idx:end_idx]
        
        z = np.dot(x_batch, thetas)
        h = sigmoid(z)
        gradient = np.dot(x_batch.T, (h - y_batch)) / batch_size
        thetas -= LEARNING_RATE * gradient
        start_idx += batch_size
    return thetas

def stochastic_gradient_decent(x: pd.DataFrame, y: pd.Series):
    thetas = np.zeros(x.shape[1])
    for _ in range(ITERATIONS):
        random_i = np.random.choice(len(x))
        new_x = x.iloc[random_i]
        z = np.dot(new_x, thetas)
        h = sigmoid(z)
        gradient = np.dot(new_x, (h - y[random_i]))
        thetas -= LEARNING_RATE * gradient
    return thetas

def get_thetas(house: str, x: pd.DataFrame, y: pd.Series, algorithm: str = 'batch'):
    # Change house value either 1 or 0
    y = y.apply(lambda h: 1 if h == house else 0)
    # Calculate thetas
    if algorithm == 'mini-batch':
        thetas = mini_batch_gradient_decent(x, y)
    elif algorithm == 'stochastic':
        thetas = stochastic_gradient_decent(x, y)
    else:
        thetas = batch_gradient_decent(x, y)
    return thetas

def predict(x: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    houses = weights.columns
    possibilities = np.zeros(len(houses))
    y = np.empty(x.shape[0], dtype=object)
    
    i = 0
    for features in x.values:
        for idx, house in enumerate(houses):
            possibilities[idx] = sigmoid(np.dot(features, weights[house]))
        # possibilities = [poss1, poss2, poss3, poss4]
        y[i] = houses[np.argmax(possibilities)]
        i += 1
    return y

def normalize(df: pd.DataFrame):
    min_values = df.min()
    max_values = df.max()
    normalized_df = (df - min_values) / (max_values - min_values)
    df[:] = normalized_df.values
    return df

def standardize(df: pd.DataFrame):
    mean = df.apply(math_mean)
    std = df.apply(math_std)
    standardized_df = (df - mean) / std
    df[:] = standardized_df.values
    return df.fillna(df.apply(math_mean))

def accuracy(y1: np.ndarray, y2: np.ndarray) -> float:
    return (y1 == y2).sum() / len(y1)

def insert_bias(x: pd.DataFrame) -> pd.DataFrame:
    x.insert(0, 'Bias', [1] * x.shape[0])
    return x