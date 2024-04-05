import argparse
import pandas as pd
from lib.validate import is_csv
from lib.print import success, info, danger
from lib.logreg import get_thetas, predict, zscore_normalize, accuracy, insert_bias, ITERATIONS
from lib.validate import is_train_dataset
import matplotlib.pyplot as plt

def check_train_dataset(df: pd.DataFrame):
    if not is_train_dataset(df):
        print(f"{danger('Error: Dataset is not train dataset.')}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="logreg_train: train a logistic regression model")
    parser.add_argument("dataset_train", type=is_csv, help="Path to the .csv file")
    parser.add_argument('-a', '--algorithm', choices=['batch', 'stochastic', 'mini-batch'], default='batch', help="Algorithm: 'batch', 'stochastic' or 'mini-batch'")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_train, index_col="Index")
    check_train_dataset(df)

    # Homogeneous     : Arithmancy, Potions, Care of Magical Creatures
    # Similarity      : (Astronomy vs Defense Against the Dark Arts), (Transfiguration vs History of Magic vs Flying)
    df.drop(columns=['Arithmancy', 'Astronomy', 'Potions', 'Care of Magical Creatures', 'Transfiguration', 'Flying'], inplace=True)
    
    # Select numeric features
    x = insert_bias(zscore_normalize(df.select_dtypes(include='number')))
    y = df['Hogwarts House']

    houses = y.unique()
    weights = pd.DataFrame(columns=houses)
    weights.index.name = 'Thetas'

    print(info(f'Start training...'))
    print(info(f"Algorithm: {args.algorithm}"))
    plt.figure(figsize=(15, 5)) 
    for house in houses:
        thetas, losses = get_thetas(house, x, y, args.algorithm)
        weights[house] = thetas
        plt.plot(range(0, ITERATIONS), losses, label=house)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function by Epoch')
    plt.savefig(f'train-{args.algorithm}.png')


    # Save weights
    weights.to_csv('weights.csv')
    print(f"Weights saved into {success('./weights.csv')}")
    print(info('Training done!'))

    # Test with train data
    print(info("\nTesting with train data..."))
    y2 = predict(x, weights)

    # Accuracy
    print(f"Accuracy: {success('{:.2%}'.format(accuracy(y.values, y2)))}")