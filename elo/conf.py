# -*- coding: utf-8 -*-

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
