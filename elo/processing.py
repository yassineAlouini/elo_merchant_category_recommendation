import pandas as pd

# Merge on card_id: train/test and historical_transactions
# Merge on merchant_id: train/test and merchants


# TODO: Refactor the train and test functions (very similar).
def get_train_df(path=None):
    # TODO: Add some documentation
    if path is not None:
        try:
            # TODO: Consider saving into Praquet file.
            df = pd.read_csv(path)
            print(df.columns)
            print(df.dtypes)
        except FileNotFoundError:
            train_df = pd.read_csv('elo/data/train.csv')
            historical_transactions_df = pd.read_csv('elo/data/historical_transactions.csv')
            print(historical_transactions_df.columns)
            merchants_df = pd.read_csv('elo/data/merchants.csv')
            print(merchants_df.columns)
            df = (train_df.merge(historical_transactions_df, on="card_id", how="left")
                          .merge(merchants_df, on="merchant_id", how="inner",
                                 suffixes=["_transactions", "_merchants"]))
            df.to_csv(path, index=False)
    return df


def get_test_df(path=None):
    """Notice that the number of card_id at the end should be the same as for the test_df."""
    if path is not None:
        try:
            # TODO: Consider saving into Praquet file.
            df = pd.read_csv(path)
            print(df.columns)
            print(df.dtypes)
        except FileNotFoundError:
            test_df = pd.read_csv('elo/data/test.csv')
            new_merchant_transactions_df = pd.read_csv('elo/data/new_merchant_transactions.csv')
            merchants_df = pd.read_csv('elo/data/merchants.csv')
            print(merchants_df.columns)
            df = (test_df.merge(new_merchant_transactions_df, on="card_id", how="left")
                         .merge(merchants_df, on="merchant_id", how="inner",
                                suffixes=["_transactions", "_merchants"]))
            df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = get_test_df('elo/data/merged_test.csv')
    print(df.sample(5).T)
