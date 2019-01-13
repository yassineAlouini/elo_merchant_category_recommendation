# -*- coding: utf-8 -*-


from pathlib import Path

import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import KFold, cross_val_score
from tpot import TPOTRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from elo.processing import get_train_df

SEED = 314
MAX_EVALS = 1000
CV = KFold(5, random_state=SEED)
# TODO: Make these bigger later.
TPOT_GENERATIONS = 10
TPOT_POPULATION_SIZE = 10


class HPOptimizer(object):

    def __init__(self):
        df = pd.read_csv('data/train.csv')
        self.X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
        self.y = df.loc[:, ["target"]]
        self.seed = SEED
        self.trials = Trials()
        self.pbar = tqdm(total=MAX_EVALS, desc="Hyperopt")

    def score(self, params):
        params["n_estimators"] = int(params["n_estimators"])
        model = XGBRegressor(**params)
        neg_mse_cv = cross_val_score(model, X=self.X, y=self.y, cv=CV, scoring="neg_mean_squared_error")
        loss = ((-1 * neg_mse_cv) ** 0.5).mean()
        print("RMSE for CV: ", loss)
        self.pbar.update()
        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'tree_method': 'exact',
            'seed': self.seed
        }
        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(self.score, space, algo=tpe.suggest,
                    # trials=trials,
                    max_evals=MAX_EVALS, trials=self.trials)
        return best


def tpot(use_dask):
    # TODO: Add some documentation...
    if use_dask:
        client = Client()
    tpot_reg = TPOTRegressor(generations=TPOT_GENERATIONS, population_size=TPOT_POPULATION_SIZE,
                             random_state=SEED,  cv=CV, use_dask=use_dask,
                             verbosity=2, memory="auto", warm_start=True, njobs=-1)
    df = get_train_df("elo/data/merged_train.csv")
    print(df.head(1))
    # TODO: Drop these categorical for now, will transform them later.
    to_drop_cols = ["first_active_month", "authorized_flag", "category_1_transactions",
                    "category_3", "merchant_id", "purchase_date", "category_1_merchants",
                    "most_recent_sales_range", "most_recent_purchases_range", "category_4"]
    # TODO: the "mean" aggregation isn't probably the best option for all the columns.
    df = (df.drop(to_drop_cols, axis=1)
            .groupby("card_id")
            .mean()
            .reset_index())
    # TODO: Find a better way to impute inf and missing values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[~df.isna()]
    print(df.shape)
    X = df.drop(["card_id", "target"], axis=1)
    y = df.loc[:, "target"]

    if use_dask:
        with ProgressBar():
            tpot_reg.fit(X, y)
    else:
        tpot_reg.fit(X, y)
    export_path = str(Path('elo/data/tpot.py').absolute())
    tpot_reg.export(export_path)


def hyperopt_xgboost():
    """ XGBoost model with hp optimized using hyperopt. """
    optimizer = HPOptimizer()
    optimal_hp = optimizer.optimize()
    print(optimizer.trials)
    model = XGBRegressor(**optimal_hp)
    neg_mse_cv = cross_val_score(model, X=optimizer.X, y=optimizer.y, cv=CV, scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    model.fit(optimizer.X, optimizer.y)
    return model


def lightGBM():
    # Finish implementing this.
    pass

# TODO: Refactor with the other function.


def make_submission(model, model_name):
    df = pd.read_csv('data/test.csv')
    X = df.loc[:, ["feature_1", "feature_2", "feature_3"]]
    df["target"] = model.predict(X)
    (df.loc[:, ["card_id", "target"]]
       .to_csv(model_name + "_submission.csv", index=False))


if __name__ == "__main__":
    # model = hyperopt_xgboost()
    # make_submission(model, "simple_hyperopt_xgboost")
    tpot(use_dask=True)
