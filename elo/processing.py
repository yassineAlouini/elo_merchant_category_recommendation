import pandas as pd

# Merge on card_id: train/test and historical_transactions
# Merge on merchant_id: train/test and merchants


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
            df = (train_df.merge(historical_transactions_df, on="card_id", how="inner")
                          .merge(merchants_df, on="merchant_id", how="inner",
                                 suffixes=["_transactions", "_merchants"]))
            df.to_csv(path, index=False)
    return df


def get_test_df():
    test_df = pd.read_csv('elo/data/test.csv')
    new_merchant_period_df = pd.read_csv('elo/data/new_merchant_transactions.csv')
    merchants_df = pd.read_csv('elo/data/merchants.csv')
    return (test_df.merge(new_merchant_period_df, on="card_id", how="inner")
                   .merge(merchants_df, on="merchant_id", how="inner"))


if __name__ == "__main__":
    df = get_train_df('elo/data/merged_train.csv')
    print(df.sample(5).T)
