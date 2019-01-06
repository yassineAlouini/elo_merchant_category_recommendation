# A linear model as a benchmark.
# with only feature_1, feature_2, and feature_3 as features.


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# TODO: Move to a conf file.
CV = KFold(5, random_state=314)


def linear_benchmark():
    df = pd.read_csv('data/train.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    y = df.loc[:, ["target"]]
    lr = LinearRegression()
    neg_mse_cv = cross_val_score(lr, X=X, y=y, cv=CV, scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    # Should be: [ 3.83557931  3.83180885  3.82162575  3.84073593  3.91959152]
    lr.fit(X, y)
    return lr


def make_benchmark_submission():
    lr = linear_benchmark()
    df = pd.read_csv('data/test.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    df["target"] = lr.predict(X)
    (df.loc[:, ["card_id", "target"]]
       .to_csv("linear_benchmark_submission.csv", index=False))
