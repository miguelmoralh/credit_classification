from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np

def get_model(trial, model_type, cat_features):
    """
    Suggests and returns a model based on the specified `model_type` and hyperparameters.
    
    Arguments:
    - trial: An Optuna trial object for hyperparameter optimization.
    - model_type: A string specifying the type of model to create ('decision_tree', 'random_forest', 'xgboost', 'catboost', 'lightgbm').
    - cat_features: A list of categorical features (only used for CatBoost).

    Returns:
    - An instantiated model with hyperparameters suggested from the Optuna trial.
    
    The function creates different machine learning models such as Decision Trees, Random Forests, XGBoost, CatBoost, and LightGBM.
    For each model, it tunes various hyperparameters using `trial.suggest_*` methods and returns the instantiated model.
    """
    if model_type == "decision_tree":
        # Create and return a DecisionTreeClassifier with suggested hyperparameters
        max_depth = trial.suggest_int("max_depth", 2, 40) 
        min_samples_split = trial.suggest_int("min_samples_split", 2, 32) 
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) 
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy']) 
        splitter = trial.suggest_categorical('splitter', ['best', 'random']) 
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            splitter=splitter,
            random_state=42
        )
    elif model_type == "random_forest":
        # Create and return a RandomForestClassifier with suggested hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 100, 1000) 
        max_depth = trial.suggest_int("max_depth", 5, 30) 
        criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]) 
        min_samples_split = trial.suggest_int("min_samples_split", 2, 32) 
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32) 
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) 
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
    elif model_type == "xgboost":
        # Create and return an XGBClassifier with suggested hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 1000, 3000) 
        max_depth = trial.suggest_int("max_depth", 1, 10) 
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1) 
        subsample = trial.suggest_float('subsample', 0.1, 1.0, log=True) 
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True) 
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20) 
        alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            alpha=alpha,
            objective="multi:softprob",  # Set objective for multiclass classification
            num_class=3 # Set number of classes 
        )
        
    elif model_type == "catboost":
        # Create and return a CatBoostClassifier with suggested hyperparameters
        iterations = trial.suggest_int("iterations", 1000, 3000) 
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True) 
        depth = trial.suggest_int("depth", 1, 10) 
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.1, 1.0, log=True)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 100) 
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ["Bernoulli", "Bayesian"]) 
        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0, 10) 
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                colsample_bylevel=colsample_bylevel,
                min_data_in_leaf=min_data_in_leaf,
                bootstrap_type=bootstrap_type, 
                bagging_temperature=bagging_temperature,
                random_state=42,
                cat_features=cat_features,
                objective='MultiClass',  # Specify the objective for multiclass classification
                verbose=0 
            )
            
        elif bootstrap_type == "Bernoulli":
            subsample = trial.suggest_float('subsample', 0.1, 1.0, log=True) 
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                colsample_bylevel=colsample_bylevel,
                min_data_in_leaf=min_data_in_leaf,
                bootstrap_type=bootstrap_type, 
                subsample=subsample,
                random_state=42,
                cat_features=cat_features,
                objective='MultiClass',  # Specify the objective for multiclass classification
                verbose=0 
            )
    elif model_type == "lightgbm":
        # Create and return an LGBMClassifier with suggested hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 1000, 3000)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True) 
        num_leaves = trial.suggest_int("num_leaves", 2, 2**10) 
        subsample = trial.suggest_float('subsample', 0.1, 1, log=True) 
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0, log=True) 
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 100) 
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True) 
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)  
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            bagging_freq = 1, 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,
            min_data_in_leaf = min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            random_state=42,
            verbosity=-1,
            objective="multiclass", # Set objective for multiclass classification
            num_class=3 
        )
    
def objective(trial, model_type, X, y, imputer, encoder, scorer, scv):
    """
    Objective function for Optuna to optimize model performance.

    Arguments:
    - trial: An Optuna trial object for hyperparameter optimization.
    - model_type: A string specifying the type of model to create ('decision_tree', 'random_forest', 'xgboost', 'catboost', 'lightgbm').
    - X: Features (input data) as a pandas DataFrame.
    - y: Labels (target values) as a pandas Series.
    - imputer: A scikit-learn transformer to handle missing values.
    - encoder: A scikit-learn transformer to encode categorical features.
    - scorer: A custom scoring function (e.g., ROC AUC) to evaluate the model.
    - scv: Stratified K-Folds cross-validator.

    Returns:
    - mean_score: The average validation score across cross-validation folds.
    - mean_overfitting_score: The average overfitting score, which is the difference between training and validation scores.
    
    The function performs cross-validation on a pipeline that includes preprocessing (imputer, encoder) and the model.
    It calculates both the validation score and overfitting (difference between train and validation scores).
    """
    model = get_model(trial, model_type, cat_features=None)
    
    # Define pipeline for preprocessing and modeling
    pipe = Pipeline([
        ('imputer', imputer),
        ('encoder', encoder),
        ('model', model)
    ])
    
    # Lists to store validation scores and overfitting measures
    scores_val_list = []
    scores_overfitting_list = []
    
    # Stratified cross validation
    for train_index, val_index in scv.split(X.copy(), y.copy()):
        # Split data into training and validation sets
        X_train, X_val = X.copy().iloc[train_index], X.copy().iloc[val_index]
        y_train, y_val = y.copy().iloc[train_index], y.copy().iloc[val_index]
        
        # Fit the pipeline and predict probabilities
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict_proba(X_train)
        y_pred_test = pipe.predict_proba(X_val)
        
        # Calculate scores for training and validation sets
        score_train = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)
        score_val = scorer._score_func(y_val, y_pred_test, **scorer._kwargs)
        
        # Calculate overfitting (difference between train and validation scores)
        overfitting_score = score_train - score_val
        scores_val_list.append(score_val)
        scores_overfitting_list.append(overfitting_score)
    
    # Return mean validation score and mean overfitting score
    mean_score = np.mean(scores_val_list)
    mean_overfitting_score = np.mean(scores_overfitting_list)
    
    return mean_score, mean_overfitting_score

def objective_categorical(trial, model_type, X, y, imputer, scorer, scv, cat_features): 
    """
    Objective function specifically for CatBoost models (which handle categorical features internally).

    Arguments:
    - trial: An Optuna trial object for hyperparameter optimization.
    - model_type: A string specifying the type of model to create (should be 'catboost').
    - X: Features (input data) as a pandas DataFrame.
    - y: Labels (target values) as a pandas Series.
    - imputer: A scikit-learn transformer to handle missing values.
    - scorer: A custom scoring function (e.g., ROC AUC) to evaluate the model.
    - scv: Stratified K-Folds cross-validator.
    - cat_features: List of categorical features (used only for CatBoost).
    
    Returns:
    - mean_score: The average validation score across cross-validation folds.
    - mean_overfitting_score: The average overfitting score, which is the difference between training and validation scores.
    
    Similar to the general `objective` function, but tailored for CatBoost models, which do not require categorical encoding.
    It handles missing data imputation but skips encoding of categorical features.
    """
    model = get_model(trial, model_type, cat_features)
    
    # Lists to store validation scores and overfitting measures
    scores_val_list = []
    scores_overfitting_list = []
    
    # Stratified cross-validation
    for train_index, val_index in scv.split(X.copy(), y.copy()):
         # Split data into training and validation sets
        X_train, X_val = X.copy().iloc[train_index], X.copy().iloc[val_index]
        y_train, y_val = y.copy().iloc[train_index], y.copy().iloc[val_index]
        
        # Impute missing values
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_val)
        
         # Train the CatBoost model and predict probabilities
        model.fit(X_train_imputed, y_train)
        y_pred_train = model.predict_proba(X_train_imputed)
        y_pred_test = model.predict_proba(X_test_imputed)
        
        # Calculate scores for training and validation sets
        score_train = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)
        score_val = scorer._score_func(y_val, y_pred_test, **scorer._kwargs)
        
        # Calculate overfitting (difference between train and validation scores)
        overfitting_score = score_train - score_val
        scores_val_list.append(score_val)
        scores_overfitting_list.append(overfitting_score)

    # Return mean validation score and mean overfitting score
    mean_score = np.mean(scores_val_list)
    mean_overfitting_score = np.mean(scores_overfitting_list)
    
    return mean_score, mean_overfitting_score