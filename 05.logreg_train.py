import argparse
from lib.validate import is_csv
import pandas as pd
from lib.print import success, info, danger
from lib.logreg import get_thetas, predict, standardize, accuracy, insert_bias
from lib.validate import is_train_dataset

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

    df.drop(columns=['Arithmancy', 'Astronomy', 'Potions', 'Care of Magical Creatures', 'Transfiguration'], inplace=True)
    
    # Select numeric features
    x = insert_bias(standardize(df.select_dtypes(include='number')))
    y = df['Hogwarts House']

    houses = y.unique()
    weights = pd.DataFrame(columns=houses)
    weights.index.name = 'Thetas'

    print(info(f'Start training...'))
    print(info(f"Algorithm: {args.algorithm}"))
    for house in houses:
        thetas = get_thetas(house, x, y, args.algorithm)
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