# ELO Merchant Category Recommendation Challenge


## Time spent


6-1-2019: 2 hours
7-1-2019: 0.5 hour
8-1-2019: 0.5 hour
12-1-2019: 3 hours
13-1-2019: 2 hours



## Benchmarks

Linear regression, 3 features (`feature_1`, `feature_2`, `feature_3`), no transformation, 5-fold CV RMSE:
[3.83557931  3.83180885  3.82162575  3.84073593  3.91959152]


## Models

Various models tried:

* XGBoost with hyperparameters optimized using hyperopt but "simple" features: nothing interesting!
* LigthGBM: not yet.
* TPOT with more features (merging merchants and transactions CSVs): in progress.

## EDA insights



## Dask-labextension installation

Activate a virtualenv, then run the following command:

```bash
conda install nodejs
pip install dask_labextension
jupyter labextension install dask-labextension

```
