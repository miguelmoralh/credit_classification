'''
Optuna uses a smart technique called Bayesian optimization to find the best hyperparameters for your model.
Bayesian optimization is like a treasure hunter using an advanced metal detector to find hidden gold, 
instead of just digging random holes (random search) or going through the entire area with a shovel (grid search).

'''

from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, TARGET_MAPPING
)
from utils.utils import match_features
from utils.utils_hyp_opt import save_pareto_front_plot, save_study_trials_to_json
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
from utils.utils_hyperparams_selection import objective, objective_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import NSGAIISampler
import functools

# Load the dataset into a pandas DataFrame
df = pd.read_csv('data/train.csv')

# Separate the target from the dataframe
x = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Encode the target variable 'y' using the provided mapping
y_encoded = y.map(TARGET_MAPPING)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=4)

### DATASET CLEANNING

data_cleaner = DataCleanning(
    object_to_float=OBJECT_TO_FLOAT,                      
    object_to_int=OBJECT_TO_INT,
    time_to_numeric_years=TIME_TO_NUMERIC_YEARS, 
    months_to_numeric=MONTHS_TO_NUMERIC,
    multi_label_binarizer_features=MULTI_LABEL_BINARIZER_FEATURES,
    nan_threshold = 0.9,
    ids_threshold = 0.12, 
    unique_threshold = 1
)

# Clean the training and test data
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

# Get the path of the stored selected features in 02_main_rfe_fs.py 
file_path = "logs/selected_features/rfe_selected_features.txt"

# Open the file and read the selected feature names
with open(file_path, 'r') as file:
    features_list = [line.strip() for line in file.readlines()]

# Filter the common columns between features_list and X_train_cleaned
original_features = match_features(features_list, X_train_cleaned)

# Filter the dataset with selected features
X_train_filtered = X_train_cleaned[original_features]
X_test_filtered = X_test_cleaned[original_features]

# Identify categorical features in the filtered training dataset
categorical_features = X_train_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

### HYPERPARAMETERS OPTIMIZATION using OPTUNA

# Initialize imputer and encoder
imputer = ImputeMissing()  
encoder = CategoricalEncoder(
    ordinal_variables=ORDINAL_VARIABLES, 
    categorical_variables=CATEGORICAL_NON_ORDINAL_VARIABLES
)

# Define the scorer for multiclass ROC AUC and the stratified cross-validation scheme
scorer_roc_auc = make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True)
scv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create separate multi-objective optimizers (ROC AUC maximized, overfitting minimized) studies for each model 
study_dt = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())
study_rf = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())
study_xgb = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())
study_cb = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())
study_lgbm = optuna.create_study(directions=["maximize", "minimize"], sampler=NSGAIISampler())

## DECISION TREE

# Define the objective function for Decision Tree with additional arguments using functools.partial
objective_dt = functools.partial(
    objective, 
    model_type = "decision_tree", 
    X=X_train_filtered.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study_dt.optimize(objective_dt, n_trials=30, show_progress_bar=True) # Optimize the study for Decision Tree
save_study_trials_to_json(study_dt, "decision_tree") # Save the optimization results for Decision Tree

## RANDOM FOREST

# Define the objective function for Random Forest with additional arguments using functools.partial
objective_rf = functools.partial(
    objective, 
    model_type = "random_forest", 
    X=X_train_filtered.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=True) # Optimize the study for Random Forest
save_study_trials_to_json(study_rf, "random_forest") # Save the optimization results for Random Forest
save_pareto_front_plot(study_rf, "random_forest") # Plot and save the Pareto front for Random Forest

## XGBOOST

# Define the objective function for XGBoost with additional arguments using functools.partial
objective_xgb = functools.partial(
    objective, 
    model_type = "xgboost", 
    X=X_train_filtered.copy(), 
    y=y_train.copy(), 
    imputer=imputer, 
    encoder=encoder, 
    scorer=scorer_roc_auc, 
    scv=scv
)
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True) # Optimize the study for XGBoost
save_study_trials_to_json(study_xgb, "xgboost") # Save the optimization results for XGBoost

## CATBOOST

# Define the objective function for CatBoost with additional arguments using functools.partial
if not categorical_features: # If no categorical features exist
    objective_cb = functools.partial(
            objective, 
            model_type = "catboost", 
            X=X_train_filtered.copy(), 
            y=y_train.copy(), 
            imputer=imputer, 
            encoder=encoder, 
            scorer=scorer_roc_auc, 
            scv=scv
    )
else: # If there are categorical features
    objective_cb = functools.partial(
        objective_categorical, 
        model_type = "catboost", 
        X=X_train_filtered.copy(), 
        y=y_train.copy(), 
        imputer=imputer,  
        scorer=scorer_roc_auc, 
        scv=scv,
        cat_features=categorical_features # Specify the categorical features
    )
study_cb.optimize(objective_cb, n_trials=30, show_progress_bar=True) # Optimize the study for CatBoost
save_study_trials_to_json(study_cb, "catboost") # Save the optimization results for CatBoost

## LIGHTGBM

# Define the objective function for LightGBM with additional arguments using functools.partial
objective_lgbm = functools.partial(
    objective, 
    model_type = "lightgbm", 
    X=X_train_filtered.copy(), 
    y=y_train.copy(), 
    imputer=imputer,
    encoder=encoder,  
    scorer=scorer_roc_auc, 
    scv=scv
)
study_lgbm.optimize(objective_lgbm, n_trials=30, show_progress_bar=True) # Optimize the study for LightGBM
save_study_trials_to_json(study_lgbm, "lightgbm") # Save the optimization results for LightGBM
save_pareto_front_plot(study_lgbm, "lightgbm") # Plot and save the Pareto front for LightGBM




