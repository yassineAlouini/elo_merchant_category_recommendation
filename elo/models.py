# -*- coding: utf-8 -*-


import os
from pathlib import Path

import numpy as np
import pandas as pd
from comet_ml import Experiment
from dask.diagnostics import Profiler, ProgressBar
from dask.distributed import Client
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.mongoexp import MongoTrials
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from tpot import TPOTRegressor
from tpot.builtins import StackingEstimator
from tqdm import tqdm
from vecstack import stacking
from xgboost import XGBClassifier, XGBRegressor

from elo.conf import (COMET_ML_API_KEY, EARLY_STOPPING_ROUNDS, FEATS_EXCLUDED,
                      HYPEROPT_LIGHTGBM_OPTIMAL_HP,
                      HYPEROPT_XGBOOST_NO_OUTLIERS_OPTIMAL_HP,
                      HYPEROPT_XGBOOST_OPTIMAL_HP,
                      HYPEROPT_XGBOOST_OUTLIERS_CLASSIFIER_OPTIMAL_HP,
                      MAX_EVALS, N_FOLDS, OPTUNA_LIGTHGBM_OPTIMAL_HP,
                      PROJECT_NAME, SEED, TPOT_GENERATIONS,
                      TPOT_POPULATION_SIZE)
from elo.processing import get_test_df, get_train_df, rmse

CV = KFold(N_FOLDS, random_state=SEED, shuffle=True)


# create an experiment with your api key
exp = Experiment(api_key=COMET_ML_API_KEY,
                 project_name=PROJECT_NAME,
                 auto_param_logging=True)


class HPOptimizer(object):

    def __init__(self, model_type="xgboost", debug=False, X=None, y=None, max_evals=MAX_EVALS):
        self.model_type = model_type

        if X is None:
            df = pd.read_csv("elo/data/augmented_train.csv")
            print(df.sample(5))
            # TODO: Find a better way to impute inf and missing values.
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median())
            X = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
            if self.model_type == "xgboost_classifier":
                y = df.loc[:, "outliers"].values
            else:
                y = df.loc[:, "target"].values
            if debug:
                X = X[:100]
                y = y[:100]
        self.X = X
        self.y = y
        self.seed = SEED
        self.trials = Trials()
        self.pbar = tqdm(total=MAX_EVALS, desc="Hyperopt for {}".format(self.model_type))
        self.max_evals = max_evals

        if self.model_type == "xgboost":
            self.model_class = XGBRegressor
        elif self.model_type == "lightgbm":
            self.model_class = LGBMRegressor
        elif self.model_type == "xgboost_classifier":
            self.model_class = XGBClassifier

    def score(self, params):
        model = self.model_class(**params)
        if self.model_type == "xgboost_classifier":
            # Trying
            loss_cv = cross_val_score(model, X=self.X, y=self.y, cv=CV,
                                      scoring="neg_log_loss")
        else:
            loss_cv = cross_val_score(model, X=self.X, y=self.y, cv=CV,
                                      scoring="neg_mean_squared_error")
        # Try a different loss for HP optimization? => in progress.
        if self.model_type == "xgboost_classifier":
            loss = (-1 * loss_cv).mean()
        else:
            loss = ((-1 * loss_cv).mean() ** 0.5)
        print("Loss for choosing hp: ", loss)
        self.pbar.update()
        exp.log_parameters(params)
        exp.log_metrics({"loss": loss, "loss_cv": loss_cv.tolist()})

        return {'loss': loss, 'status': STATUS_OK}

    def optimize(self):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here), finds the best hyperparameters.
        """
        # XGBoost space
        if self.model_type == "xgboost" or self.model_type == "xgboost_classifier":
            space = {
                'n_estimators': 100 + hp.randint('n_estimators', 1000),
                'eta': hp.loguniform('eta', np.log(1e-5), np.log(1.0)),
                # A problem with max_depth casted to float instead of int with
                # the hp.quniform method.
                # TODO: Was too big, trying smaller for now (maybe it caused bad malloc?)
                'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
                'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                # TODO: Try GPU later.
                'tree_method': 'hist',
                'seed': self.seed,
                'n_jobs': 8
            }
        if self.model_type == "xgboost_classifier":
            print((self.y == 0).sum() / (self.y == 1).sum())
            space["scale_pos_weight"] = (self.y == 0).sum() / (self.y == 1).sum()
        else:
            # LigthGBM space
            # TODO: Try other boosting types later.
            # Smaller n_estimtors.
            space = {'boosting_type': 'goss',
                     'n_estimators': 100 + hp.randint('n_estimators', 400),
                     'learning_rate': hp.loguniform('learning_rate', np.log(1e-8), np.log(1.0)),
                     'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
                     'num_leaves': hp.choice('num_leaves', [15, 31, 63, 127, 200]),
                     'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
                     'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
                     'min_child_weight': hp.quniform('min_child_weight', 1, 100, 1),
                     'reg_alpha': hp.quniform('reg_alpha', 1, 100, 1),
                     'min_split_gain': hp.quniform('min_split_gain', 1, 100, 1),
                     'min_data_in_leaf':  10 + hp.randint('min_data_in_leaf', 100),
                     'bagging_seed': self.seed,
                     'drop_seed': self.seed,
                     'seed': self.seed}

        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(self.score, space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials)
        return best


def tpot(use_dask=True):
    # TODO: Add some documentation...
    # TODO: Investigate why tpot crashes when uing Dask (probably a RAM problem).
    if use_dask:
        client = Client("tcp://192.168.1.94:8786")
        print(client)
    tpot_reg = TPOTRegressor(generations=TPOT_GENERATIONS, population_size=TPOT_POPULATION_SIZE,
                             random_state=SEED,  cv=CV, use_dask=use_dask,
                             verbosity=2, memory="auto")
    df = pd.read_csv("elo/data/augmented_train.csv")
    print(df.sample(5))
    # TODO: Find a better way to impute inf and missing values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    X = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    y = df.loc[:, "target"].values

    if use_dask:
        with ProgressBar() as pbar, Profiler() as prof:
            tpot_reg.fit(X, y)
    else:
        tpot_reg.fit(X, y)
    export_path = str(Path('elo/data/tpot_few_generations_augmented_dataset.py').absolute())
    tpot_reg.export(export_path)
    return tpot_reg


def best_tpot_few_generations():
    """ The output of the TPOT pipeline with 10 generations and 10 population size. """
    model = make_pipeline(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.01, loss="quantile",
                                                              max_depth=2,
                                                              max_features=0.9500000000000001, min_samples_leaf=20,
                                                              min_samples_split=20, n_estimators=100, subsample=0.55)),
        ExtraTreesRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=17,
                            min_samples_split=14, n_estimators=100)
    )
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
    df = df.fillna(df.median())
    print(df.shape)
    X = df.drop(["card_id", "target"], axis=1)
    y = df.loc[:, "target"]
    model.fit(X, y)
    return model


def hyperopt_model(model_type, debug=False, X=None, y=None):
    """ Model with hp optimized using hyperopt. """
    optimizer = HPOptimizer(model_type=model_type, debug=debug, X=X, y=y)
    optimal_hp = optimizer.optimize()
    print(optimal_hp)
    print(optimizer.trials)
    model = optimizer.model_class(**optimal_hp)
    neg_mse_cv = cross_val_score(model, X=optimizer.X, y=optimizer.y, cv=CV,
                                 scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    model.fit(optimizer.X, optimizer.y)

    return model


def hyperopt_model_oof(model_type, model_name, debug=False, X=None, y=None):
    """ Model with hp optimized using hyperopt and OOF predictions. """
    optimizer = HPOptimizer(model_type=model_type, debug=debug, X=X, y=y)
    optimal_hp = optimizer.optimize()
    print(optimal_hp)
    model = optimizer.model_class(**optimal_hp)
    # Save trials
    try:
        with open("elo/data/{model_name}_trials.hp", "wb") as f:
            pickle.dump(optimizer.trials, f)
    except:
        pass
    test_df = pd.read_csv("elo/data/augmented_test.csv")
    # TODO: Find a better way to impute inf and missing values.
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(test_df.median())
    preds = []
    scores = []
    test_features = test_df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    for fold, (train_idx, valid_idx) in enumerate(CV.split(optimizer.X, optimizer.y)):

        model.fit(optimizer.X[train_idx], optimizer.y[train_idx])
        preds.append(model.predict(test_features))
        print(f"CV RMSE for fold {fold}")
        print(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
        scores.append(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
    print(scores)
    print(sum(scores) / len(scores))
    test_df["target"] = np.array(preds).mean(axis=0)
    (test_df.loc[:, ["card_id", "target"]]
            .to_csv(model_name + "_submission.csv", index=False))

# TODO: Add comet.ml tracking.


def hyperopt_model_oof_no_outliers(model_type, model_name, debug=False, optimal_hp=None):
    """ Model with hp optimized using hyperopt and OOF predictions without outliers. """

    exp.add_tag(model_name)

    df = pd.read_csv("elo/data/augmented_train.csv")
    df = df.replace([np.inf, -np.inf], np.nan)
    # No missing data imputation, see what happens.
    # Keep only rows without outliers.
    df = df.loc[df.outliers == 0]
    X = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    y = df.loc[:, "target"].values
    print(optimal_hp)
    optimizer = HPOptimizer(model_type=model_type, debug=debug, X=X, y=y)
    if optimal_hp is None:
        optimal_hp = optimizer.optimize()
    print(optimal_hp)
    exp.log_dataset_hash(X)
    exp.log_parameters(optimal_hp)

    model = optimizer.model_class(**optimal_hp)
    # Save trials
    try:
        with open("elo/data/{model_name}_trials.hp", "wb") as f:
            pickle.dump(optimizer.trials, f)
    except:
        pass
    test_df = pd.read_csv("elo/data/augmented_test.csv")
    # TODO: Find a better way to impute inf and missing values.
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(test_df.median())
    preds = []
    scores = []
    test_features = test_df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    for fold, (train_idx, valid_idx) in enumerate(CV.split(optimizer.X, optimizer.y)):

        model.fit(optimizer.X[train_idx], optimizer.y[train_idx])
        preds.append(model.predict(test_features))
        print(f"CV loss for fold {fold}")
        print(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
        scores.append(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
    test_df["target"] = np.array(preds).mean(axis=0)
    (test_df.loc[:, ["card_id", "target"]]
            .to_csv(model_name + "_submission.csv", index=False))


def hyperopt_model_oof_classification(model_type, model_name, debug=False, optimal_hp=None):
    """ Model with hp optimized using hyperopt and OOF predictions classification for outliers. """

    exp.add_tag(model_name)

    df = pd.read_csv("elo/data/augmented_train.csv")
    df = df.replace([np.inf, -np.inf], np.nan)
    optimizer = HPOptimizer(model_type=model_type, debug=debug)
    if optimal_hp is None:
        optimal_hp = optimizer.optimize()
    print(optimal_hp)
    exp.log_parameters(optimal_hp)

    model = optimizer.model_class(**optimal_hp)
    # Save trials
    try:
        with open("elo/data/{model_name}_trials.hp", "wb") as f:
            pickle.dump(optimizer.trials, f)
    except:
        pass
    test_df = pd.read_csv("elo/data/augmented_test.csv")
    # TODO: Find a better way to impute inf and missing values.
    test_df = test_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.fillna(test_df.median())
    preds = []
    scores = []
    test_features = test_df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    for fold, (train_idx, valid_idx) in enumerate(CV.split(optimizer.X, optimizer.y)):

        model.fit(optimizer.X[train_idx], optimizer.y[train_idx])
        preds.append(model.predict_proba(test_features))
        # print(f"CV loss for fold {fold}")
        # print(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
        # scores.append(rmse(model.predict(optimizer.X[valid_idx]), optimizer.y[valid_idx]))
    # exp.log_metrics({"global_loss": sum(scores) / len(scores), "global_loss_cv": scores})
    test_df["outliers"] = np.array(preds).mean(axis=0)
    (test_df.loc[:, ["card_id", "outliers"]]
            .to_csv(model_name + "_submission.csv", index=False))


def optuna_lightgbm():

    df = pd.read_csv("elo/data/augmented_train.csv")
    print(df.sample(5))
    # TODO: Find a better way to impute inf and missing values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    X = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    y = df.loc[:, "target"].values
    model = LGBMRegressor(**OPTUNA_LIGTHGBM_OPTIMAL_HP)
    neg_mse_cv = cross_val_score(model, X=X, y=y, cv=CV, scoring="neg_mean_squared_error")
    rmse_cv = (-1 * neg_mse_cv) ** 0.5
    print(rmse_cv)
    # TODO: fit-predict oof later.
    model.fit(X, y)
    return model


def hyperopt_lightgbm(debug=False, X=None, y=None):
    """ Ligthgbm model with hp optimized using hyperopt. """
    return hyperopt_model(model_type="lightgbm", debug=debug, X=X, y=y)


def hyperopt_xgboost(debug=False,  X=None, y=None):
    """ XGBoost model with hp optimized using hyperopt. """
    return hyperopt_model(model_type="xgboost", debug=debug, X=X, y=y)


def make_submission(model, model_name):
    df = pd.read_csv("elo/data/augmented_test.csv")
    # TODO: Find a better way to impute inf and missing values.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    X = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
    df["target"] = model.predict(X)
    (df.loc[:, ["card_id", "target"]]
       .to_csv(model_name + "_submission.csv", index=False))


if __name__ == "__main__":
    hyperopt_model_oof_classification(
        "xgboost_classifier",
        "hyperopt_xgboost_more_folds_more_iterations_shuffling_oof_outliers_classification_augmented_dataset",
        optimal_hp=HYPEROPT_XGBOOST_OUTLIERS_CLASSIFIER_OPTIMAL_HP)
