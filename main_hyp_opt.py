from constant_variables import (
    object_to_float, object_to_int, time_to_numeric_years, months_to_numeric,
    multi_label_binarizer_features, ordinal_variables, 
    categorical_non_ordinal_variables
)
from utils.data_cleanning import DataCleanning
from utils.imputer import ImputeMissing
from utils.cateorical_encoder import CategoricalEncoder
from hyperparams_optimization import objective, get_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
import functools

# Load dataframe
df = pd.read_csv('data/train.csv')

# Separate the target from the dataframe
x = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

### DATASET CLEANNING
data_cleaner = DataCleanning(
    object_to_float=object_to_float,                      
    object_to_int=object_to_int,
    time_to_numeric_years=time_to_numeric_years, 
    months_to_numeric=months_to_numeric,
    multi_label_binarizer_features=multi_label_binarizer_features,
    nan_threshold = 0.9,
    ids_threshold = 0.12, 
    unique_threshold = 1
)
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

# Initialize imputer and encoder
imputer = ImputeMissing()  
encoder = CategoricalEncoder(
    ordinal_variables=ordinal_variables, 
    categorical_variables=categorical_non_ordinal_variables
)

### HYPERPARAMETERS OPTIMIZATION using OPTUNA

# Define the scorer for multiclass ROC AUC, the stratified cross validation and the multi-objective optimizer
scorer_roc_auc = make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True)
scv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Multi-objective optimizer where the score is maximized and the overfitting is minimized
study = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())

## RANDOM FOREST

# Define the objective function with additional arguments using functools.partial and Optimize
objective_rf = functools.partial(
    objective, 
    model_type = "random_forest", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_rf, n_trials=50, show_progress_bar=True)

## DECISION TREE
objective_dt = functools.partial(
    objective, 
    model_type = "decision_tree", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_dt, n_trials=30, show_progress_bar=True)

## ADABOOST
objective_ab = functools.partial(
    objective, 
    model_type = "adaboost", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_ab, n_trials=30, show_progress_bar=True)

## XGBOOST
objective_xgb = functools.partial(
    objective, 
    model_type = "xgboost", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

## CATBOOST
objective_cb = functools.partial(
    objective, 
    model_type = "catboost", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_cb, n_trials=30, show_progress_bar=True)

## LIGHTGBM
objective_lgbm = functools.partial(
    objective, 
    model_type = "lightgbm", 
    X=X_train_cleaned.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study.optimize(objective_lgbm, n_trials=30, show_progress_bar=True)

## HISTGRADIENTBOOSTING