import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from vecstack import stacking
from xgboost import XGBRegressor

from elo.conf import (FEATS_EXCLUDED, HYPEROPT_LIGHTGBM_OPTIMAL_HP,
                      HYPEROPT_XGBOOST_OPTIMAL_HP,
                      HYPEROPT_XGBOOST_SECOND_LEVEL_OPTIMAL_HP, N_FOLDS,
                      OPTUNA_LIGTHGBM_OPTIMAL_HP, SEED)
from elo.models import CV, hyperopt_xgboost, make_submission
from elo.processing import rmse

GRADIENT_BOOSTED_TREES_FIRST_LEVEL_PATH = "elo/data/stacking/[2019.02.02].[18.17.18].215569.1f0918.npy"
TEN_MODELS_FIRST_LEVEL_PATH = "elo/data/stacking/[2019.02.02].[21.35.10].585291.29484f.npy"
TEN_MODELS_MORE_FOLDS_SHUFFLING_FIRST_LEVEL_PATH = "elo/data/stacking/[2019.02.02].[23.44.11].261273.e813c3.npy"


def get_stacking_features(path=None):
    print(f"Training for {N_FOLDS} CV folds")
    if path is None:
        # TODO: Some refactoring.
        df = pd.read_csv("elo/data/augmented_train.csv")
        print(df.sample(5))
        # TODO: Find a better way to impute inf and missing values.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        X_train = df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
        y_train = df.loc[:, "target"].values

        test_df = pd.read_csv("elo/data/augmented_test.csv")
        print(test_df.sample(5))
        # TODO: Find a better way to impute inf and missing values.
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.fillna(test_df.median())
        X_test = test_df.drop(FEATS_EXCLUDED, axis=1, errors='ignore').values
        first_level_models = [XGBRegressor(**HYPEROPT_XGBOOST_OPTIMAL_HP),
                              LGBMRegressor(**HYPEROPT_LIGHTGBM_OPTIMAL_HP),
                              LGBMRegressor(**OPTUNA_LIGTHGBM_OPTIMAL_HP),
                              XGBRegressor(seed=SEED),
                              LGBMRegressor(seed=SEED),
                              KNeighborsRegressor(),
                              LinearRegression(),
                              ExtraTreesRegressor(random_state=SEED),
                              GradientBoostingRegressor(random_state=SEED),
                              Lasso(random_state=SEED)]
        # This didn't work at all without proper tuning!!!
        # SGDRegressor(random_state=SEED)]

        # TODO: Should I add "shuffling"?

        stacked_train, stacked_test = stacking(first_level_models, X_train, y_train, X_test,
                                               regression=True, metric=rmse, n_folds=N_FOLDS,
                                               random_state=SEED, verbose=2, save_dir="elo/data/stacking",
                                               shuffle=True)

    else:
        stacked_train, stacked_test = np.load(path)
        df = pd.read_csv("elo/data/augmented_train.csv")
        # TODO: Find a better way to impute inf and missing values.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())
        y_train = df.loc[:, "target"].values
    # Observe the data
    print(stacked_train[:5])
    print(stacked_test[:5])
    print(y_train[:5])
    return stacked_train, stacked_test, y_train


stacked_train, stacked_test, y_train = get_stacking_features(path=TEN_MODELS_MORE_FOLDS_SHUFFLING_FIRST_LEVEL_PATH)

# TODO: Try StratifiedKFold (over the "outliers" column)?
# TODO: Hyperopt this second-level model? Add CV evaluation?
# TODO: Is this the correct CV procedure for second-level models? Investiagte.
# HYPEROPT_XGBOOST_SECOND_LEVEL_OPTIMAL_HP => overfit a lot.
second_level_model = XGBRegressor(seed=SEED, tree_method="hist")
preds = []
# OOF predictions
for fold, (train_idx, valid_idx) in enumerate(CV.split(stacked_train, y_train)):

    second_level_model.fit(stacked_train[train_idx], y_train[train_idx])
    preds.append(second_level_model.predict(stacked_test))
    print(f"CV RMSE for fold {fold}")
    print(rmse(second_level_model.predict(stacked_train[valid_idx]), y_train[valid_idx]))
model_name = "stacking_ten_models_more_folds_shuffling_augmented_dataset"


test_df = pd.read_csv("elo/data/augmented_test.csv")
print(test_df.sample(5))
# TODO: Find a better way to impute inf and missing values.
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.fillna(test_df.median())

test_df["target"] = np.array(preds).mean(axis=0)
# test_df["target"] = stacked_test.mean(axis=1)
(test_df.loc[:, ["card_id", "target"]]
        .to_csv(model_name + "_submission.csv", index=False))
