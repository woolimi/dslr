import numpy as np
import pandas as pd
from lib.math import math_mean, math_std

LEARNING_RATE = 0.5
ITERATIONS = 1000

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def batch_gradient_descent(x: pd.DataFrame, y: pd.Series):
    thetas = np.zeros(x.shape[1])
    losses = []
    
    for _ in range(ITERATIONS):
        z = np.dot(x, thetas)
        h = sigmoid(z)
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h)) 
        losses.append(loss)
        gradient = np.dot(x.T, (h - y)) / y.size
        thetas -= LEARNING_RATE * gradient
    return thetas, losses

def mini_batch_gradient_descent(x: pd.DataFrame, y: pd.Series, batch_size=5):
    thetas = np.zeros(x.shape[1])
    m = len(x)
    losses = []

    for _ in range(ITERATIONS):
        random_indices = np.random.choice(m, size=batch_size, replace=False)
        x_batch = x.iloc[random_indices]
        y_batch = y.iloc[random_indices]
        
        z = np.dot(x_batch, thetas)
        h = sigmoid(z)
        loss = -np.mean(y_batch * np.log(h) + (1 - y_batch) * np.log(1 - h)) 
        losses.append(loss)
        gradient = np.dot(x_batch.T, (h - y_batch)) / batch_size
        thetas -= LEARNING_RATE * gradient
    return thetas, losses


def stochastic_gradient_descent(x: pd.DataFrame, y: pd.Series):
    thetas = np.zeros(x.shape[1])
    losses = []

    for _ in range(ITERATIONS):
        random_i = np.random.choice(len(x))
        new_x = x.iloc[random_i]
        z = np.dot(new_x, thetas)
        h = sigmoid(z)
        loss = -np.mean(y[random_i] * np.log(h) + (1 - y[random_i]) * np.log(1 - h)) 
        losses.append(loss)
        gradient = np.dot(new_x, (h - y[random_i]))
        thetas -= LEARNING_RATE * gradient
    return thetas, losses

def get_thetas(house: str, x: pd.DataFrame, y: pd.Series, algorithm: str = 'batch'):
    # Change house value either 1 or 0
    y = y.apply(lambda h: 1 if h == house else 0)
    # Calculate thetas
    if algorithm == 'mini-batch':
        thetas, losses = mini_batch_gradient_descent(x, y)
    elif algorithm == 'stochastic':
        thetas, losses = stochastic_gradient_descent(x, y)
    else:
        thetas, losses = batch_gradient_descent(x, y)

    return thetas, losses

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

def zscore_normalize(df: pd.DataFrame):
    mean = df.apply(math_mean)
    std = df.apply(math_std)
    zscore_normalized_df = (df - mean) / std
    df[:] = zscore_normalized_df.values
    return df.fillna(df.apply(math_mean))

def accuracy(y1: np.ndarray, y2: np.ndarray) -> float:
    return (y1 == y2).sum() / len(y1)

def insert_bias(x: pd.DataFrame) -> pd.DataFrame:
    x.insert(0, 'Bias', [1] * x.shape[0])
    return x