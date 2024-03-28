import argparse
import pandas as pd
from lib.validate import is_csv, is_train_dataset
from lib.logreg import insert_bias, standardize, predict
from lib.print import success, danger

def check_predict_dataset(df: pd.DataFrame):
    if is_train_dataset(df):
        print(f"{danger('Error: Dataset is not train dataset.')}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="logreg_predict: predict using a logistic regression model")
    parser.add_argument("dataset_test", type=is_csv, help="Path to the dataset_test.csv file")
    parser.add_argument("weights", type=is_csv, help="Path to the weights.csv file")

    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset_test, index_col="Index")
    df.drop(columns=['Hogwarts House', 'Arithmancy', 'Astronomy', 'Potions', 'Care of Magical Creatures', 'Transfiguration'], inplace=True)
    x = insert_bias(standardize(df.select_dtypes(include='number')))
    weights = pd.read_csv(args.weights, index_col='Thetas')

    predicted_df = pd.DataFrame(columns=['Hogwarts House'], index=df.index)
    predicted_df['Hogwarts House'] = predict(x, weights)
    predicted_df.to_csv('houses.csv')
    print(success('Predictions saved into houses.csv'))

