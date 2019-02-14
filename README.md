# ELO Merchant Category Recommendation Challenge


## Time spent


6-1-2019: 2 hours
7-1-2019: 0.5 hour
8-1-2019: 0.5 hour
12-1-2019: 3 hours
13-1-2019: 2 hours
29-1-2019: 1 hour
30-1-2019: 1 hour
31-1-2019: 1 hour
1-2-2019: 1 hour
2-2-2019: 2 hours
7-2-2019: 1 hour
9-2-2019: 1 hour


## Benchmarks

Linear regression, 3 features (`feature_1`, `feature_2`, `feature_3`), no transformation, 5-fold CV RMSE:
[3.83557931  3.83180885  3.82162575  3.84073593  3.91959152]


## Models

Various models tried:

* XGBoost with hyperparameters optimized using hyperopt but "simple" features: nothing interesting!
* LigthGBM: not yet.
* TPOT with more features (merging merchants and transactions CSVs): in progress.
* XGBoost with hyperparameters optimized using hyperopt and "augmented" features (1000 iterations):
[3.65221624 3.66138945 3.63717364 3.68112107 3.71872903].
* lightGBM with hyperparmeters optimized using hyperopt and "augmented" features (1 iteration):
[3.67962314 3.6863681  3.66617114 3.69913309 3.75430576]
* ligthGBM with hyperparmeters optimized using optuna and "augmented" features:
 [3.7026815 3.70320605 3.69626404 3.72362202 3.78065989]

* stacking the previous boosted models and "augmented" feautres:

first level (3 models):

- XGBRegressor with Hyperopt: [3.65221624 3.66138945 3.63717364 3.68112107 3.71872903]
- LGBMRegressor with Hyperopt (1 iteration): [3.67962314 3.6863681  3.66617114 3.69913309 3.75430576]
- LGBMRegressor with Optuna (from community):  [3.69012899 3.69258323 3.68182657 3.71320337 3.76940061]

second level (1 model):

- XGBRegressor (with hp optimization): [3.83573338 3.83268598 3.82201365 3.84107985 3.92014293] => overfitting problem!!!
- XGBRegressor (without): [3.64616126 3.65565184 3.63211079 3.68470507 3.71068868]
- Average:
- More models (10): [3.64990121 3.6536569  3.63004623 3.6796089  3.70409599]
- More models (10), more folds (9 instead of 5), and shuffling:
[3.600296415158376, 3.7680317104317, 3.686445101202575, 3.587128135519146, 3.562128382380808, 3.724173364737555,
 3.6358431775388302, 3.7553181821929087]


XGBoost with more folds, OOF predictions, shuffling, and MAE metric: [3.64951050451646, 3.8216460054827524, 3.7097159025940165, 3.657103964815245,
             3.6224894036411093, 3.6068337540897177, 3.767708913088468,
             3.6790960847220235, 3.8055858778764406]

- XGBoost HP optimized using hyperopt + 500 iterations + parallel execution for XGBoost (this run for a very long time, 110 hours and 35 minutes):
[3.646351548468849, 3.832613662665344, 3.7177869456354395, 3.6610552032784778, 3.6280498985629674, 3.6120725026619236, 3.7778623541798955, 3.6847024484012123, 3.81279406605563]


- No outliers model with model with outliers and classification: https://www.kaggle.com/waitingli/combining-your-model-with-a-model-without-outlier. => very promising.

- No outliers model with HP optimization using hyperopt + classification model using StratifiedKFold on the
outliers column + best stacked model: in progress.


Notice: features engineering is "inspired" (read copied) from the work of https://github.com/MitsuruFujiwara
(and probably many more).

## EDA insights



## Dask-labextension installation

Activate a virtualenv, then run the following command:

```bash
conda install nodejs
pip install dask_labextension
jupyter labextension install dask-labextension

```
