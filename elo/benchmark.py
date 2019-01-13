# A linear model as a benchmark.
# with only feature_1, feature_2, and feature_3 as features.


import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

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


def ridge_linear_benchmark():
    df = pd.read_csv('data/train.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    y = df.loc[:, ["target"]]
    # TODO: Try other values of alphas.
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 13500])
    neg_mse_cv = cross_val_score(model, X=X, y=y, cv=CV, scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    # Should be: [ 3.83553877  3.83183958  3.82161315  3.84071125  3.91960783]
    model.fit(X, y)
    print(model.alpha_)
    return model


def simple_xgboost():
    df = pd.read_csv('data/train.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    y = df.loc[:, ["target"]]
    model = XGBRegressor()
    neg_mse_cv = cross_val_score(model, X=X, y=y, cv=CV, scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    # Should be: [ 3.83417966  3.83095803  3.82135517  3.83964462  3.91841192]
    model.fit(X, y)
    return model


def make_benchmark_submission(model, model_name):
    df = pd.read_csv('data/test.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    df["target"] = model.predict(X)
    (df.loc[:, ["card_id", "target"]]
       .to_csv(model_name + "_benchmark_submission.csv", index=False))


if __name__ == "__main__":
    model = simple_xgboost()
    make_benchmark_submission(model, "simple_xgboost")
