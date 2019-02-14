# -*- coding: utf-8 -*-

import os

PROJECT_NAME = "elo_merchant_category_recommandation"
COMET_ML_API_KEY = os.environ.get("COMET_ML_API_KEY")

TRAIN_COLUMNS = ['first_active_month', 'card_id', 'feature_1', 'feature_2', 'feature_3',
                 'target']

TRANSACTIONS_COLUMNS = ['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',
                        'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',
                        'purchase_amount', 'purchase_date', 'category_2', 'state_id',
                        'subsector_id']

MERCHANTS_COLUMNS = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
                     'subsector_id', 'numerical_1', 'numerical_2', 'category_1',
                     'most_recent_sales_range', 'most_recent_purchases_range',
                     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
                     'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
                     'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12',
                     'category_4', 'city_id', 'state_id', 'category_2']

MERGED_COLUMNS = TRAIN_COLUMNS + ['authorized_flag', 'city_id_x', 'category_1_x',
                                  'installments', 'category_3', 'merchant_category_id_x', 'merchant_id',
                                  'month_lag', 'purchase_amount', 'purchase_date', 'category_2_x',
                                  'state_id_x', 'subsector_id_x', 'merchant_group_id',
                                  'merchant_category_id_y', 'subsector_id_y', 'numerical_1',
                                  'numerical_2', 'category_1_y', 'most_recent_sales_range',
                                  'most_recent_purchases_range', 'avg_sales_lag3', 'avg_purchases_lag3',
                                  'active_months_lag3', 'avg_sales_lag6', 'avg_purchases_lag6',
                                  'active_months_lag6', 'avg_sales_lag12', 'avg_purchases_lag12',
                                  'active_months_lag12', 'category_4', 'city_id_y', 'state_id_y',
                                  'category_2_y']

FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']

OUTLIER_THRESHOLD = -30

SEED = 314
# 1000 was too much, 100 is probably enough. Trying 500 => not useful it seems.
MAX_EVALS = 100
# Around 1% of the MAX_EVALS.
EARLY_STOPPING_ROUNDS = 100
# Trying more folds, let's see what happens.
N_FOLDS = 9
# TODO: Make these much bigger later.
TPOT_GENERATIONS = 100
TPOT_POPULATION_SIZE = 10


HYPEROPT_XGBOOST_OPTIMAL_HP = {'colsample_bytree': 0.6000000000000001, 'eta': 0.25, 'gamma': 0.9500000000000001,
                               'max_depth': 4, 'min_child_weight': 5.0, 'n_estimators': 121, 'subsample': 1.0}

# Try these later: (from here: https://www.kaggle.com/mfjwr1/simple-lightgbm-without-blending)
OPTUNA_LIGTHGBM_OPTIMAL_HP = {
    'learning_rate': 0.01,
    'subsample': 0.9855232997390695,
    'max_depth': 7,
    'top_rate': 0.9064148448434349,
    'num_leaves': 63,
    'min_child_weight': 41.9612869171337,
    'other_rate': 0.0721768246018207,
    'reg_alpha': 9.677537745007898,
    'colsample_bytree': 0.5665320670155495,
    'min_split_gain': 9.820197773625843,
    'reg_lambda': 8.2532317400459,
    'min_data_in_leaf': 21,
    'seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED
}

# One iteration
# RMSE for CV:  2.8628355665216805e+37
HYPEROPT_LIGHTGBM_OPTIMAL_HP = {'colsample_bytree': 0.9,
                                'learning_rate': 0.06221820285035194,
                                'max_depth': 6, 'min_child_weight': 75.0,
                                'min_data_in_leaf': 44, 'min_split_gain': 73.0,
                                'n_estimators': 1154,
                                'num_leaves': 3, 'reg_alpha': 61.0, 'subsample':
                                0.9500000000000001}

# Maybe some overfitting?
HYPEROPT_XGBOOST_SECOND_LEVEL_OPTIMAL_HP = {'colsample_bytree': 1.0, 'eta': 0.325, 'gamma': 0.55, 'max_depth': 0,
                                            'min_child_weight': 4.0, 'n_estimators': 85, 'subsample': 0.8500000000000001}


MAE_HYPEROPT_XGBOOST = {'colsample_bytree': 0.7000000000000001, 'eta': 0.6334420157484983,
                        'gamma': 0.7000000000000001, 'max_depth': 0, 'min_child_weight': 2.0,
                        'n_estimators': 115, 'subsample': 1.0}

HYPEROPT_XGBOOST_MORE_FOLDS_OPTIMAL_HP = {'colsample_bytree': 0.8500000000000001, 'eta': 0.21166208775249726,
                                          'gamma': 0.9, 'max_depth': 3, 'min_child_weight': 5.0, 'n_estimators': 37,
                                          'subsample': 0.9}


HYPEROPT_XGBOOST_MORE_FOLDS_MORE_ITERATIONS_OPTIMAL_HP = {'colsample_bytree': 0.7000000000000001,
                                                          'eta': 8.998222794660417e-05,
                                                          'gamma': 0.8500000000000001, 'max_depth': 4,
                                                          'min_child_weight': 2.0, 'n_estimators': 17,
                                                          'subsample': 0.8}
HYPEROPT_XGBOOST_NO_OUTLIERS_OPTIMAL_HP = {'colsample_bytree': 0.6000000000000001, 'eta': 0.23560138217824053,
                                           'gamma': 0.75, 'max_depth': 3, 'min_child_weight': 4.0,
                                           'n_estimators': 385, 'subsample': 1.0}
HYPEROPT_XGBOOST_OUTLIERS_CLASSIFIER_OPTIMAL_HP = {'colsample_bytree': 0.5, 'eta': 0.23838065814721812,
                                                   'gamma': 0.8500000000000001, 'max_depth': 11,
                                                   'min_child_weight': 6.0, 'n_estimators': 177, 'subsample': 1.0}
