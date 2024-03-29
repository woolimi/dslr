import pandas as pd
from lib.logreg import accuracy
from lib.print import success

if __name__ == "__main__":
    predicted_df = pd.read_csv('./houses.csv', index_col='Index')
    truth_df = pd.read_csv('./dataset_truth.csv', index_col='Index')

    print(f"Accuracy: {success('{:.2%}'.format(accuracy(predicted_df.values, truth_df.values)))}")
