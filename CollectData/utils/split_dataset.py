import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
        csv_path="datasets/dataset.csv"
):

    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_df.to_csv("datasets/train.csv", index=False)
    test_df.to_csv("datasets/test.csv", index=False)

    print("Dataset split completed")


if __name__ == "__main__":
    split_dataset()