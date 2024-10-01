from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
import xgboost as xgb
import optuna.visualization as vis
import json
import os
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


def save_study_trials_to_json(study, model_name, output_dir="logs/optuna_trials"):
    """
    Save the trials of an Optuna study to a JSON file.

    This function iterates over the trials from an Optuna study object, extracts relevant
    information (trial number, parameters, values, and state), and saves it into a JSON file.
    The file is saved in the specified output directory under the name '{model_name}_trials.json'.

    Args:
        study (optuna.Study): The Optuna study object containing trial data.
        model_name (str): The name of the model for which the trials are being saved.
        output_dir (str, optional): Directory where the JSON file should be saved. Defaults to "logs/optuna_trials".

    Returns:
        None
    """
    # Create the file path
    trials_file_path = f"{output_dir}/{model_name}_trials.json"

    # Prepare data for JSON
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "Trial Number": trial.number,
            "Params": trial.params,
            "Values": trial.values,
            "Trial State": trial.state.name  # Convert enum to string
        }
        trials_data.append(trial_data)

    # Write data to a JSON file
    with open(trials_file_path, 'w') as file:
        json.dump(trials_data, file, indent=4)

    print(f"Trials for {model_name} have been saved to {trials_file_path}")
    
def save_pareto_front_plot(study, model_name, output_dir="logs/plots"):
    """
    Plot and save the Pareto front for an Optuna study.

    This function creates a Pareto front plot for a given Optuna study, which visualizes
    the trade-off between two objectives, such as ROC AUC and overfitting. The plot is then saved as a PNG file.

    Args:
        study (optuna.Study): The Optuna study object containing trial data.
        model_name (str): The name of the model for which the Pareto front is being plotted.
        output_dir (str, optional): Directory where the plot should be saved. Defaults to "logs/plots".

    Returns:
        None
    """
    # Create the file path
    plot_file_path = f"{output_dir}/pareto_{model_name}.png"

    # Plot the Pareto front
    fig = vis.plot_pareto_front(study, target_names=["ROC AUC", "Overfitting"])

    # Save the plot
    fig.write_image(plot_file_path)

    print(f"Pareto front plot for {model_name} has been saved to {plot_file_path}")
    
def find_best_trial(logs_dir, overfitting_threshold=0.02):
    """
    Function to find the best trial across multiple models from Optuna optimization JSON logs.

    This function looks through multiple JSON files in a specified directory (each representing
    the trials of different models), checks the overfitting values against a threshold, and 
    selects the best trial based on the highest ROC AUC score while meeting the overfitting criteria.

    Args:
        logs_dir (str): Directory containing the JSON logs of Optuna trials.
        overfitting_threshold (float, optional): Maximum acceptable overfitting score. Defaults to 0.02.

    Returns:
        dict: A dictionary with information about the best trial (model name, trial number, ROC AUC, overfitting, and parameters).
    """
    best_trial = None

    # Go through all json files
    for filename in os.listdir(logs_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(logs_dir, filename)
            
            # Open the json file
            with open(filepath, 'r') as file:
                trials = json.load(file)
                
                # Extract the model name by removing '_trials.json' from the filename
                model_name = filename.replace('_trials.json', '') 
                
                # Iterate over all trials of each file
                for trial in trials:
                    roc_auc, overfitting = trial['Values']
                    
                    # If overfitting is below the threshold, check if this is the best ROC AUC so far
                    if overfitting < overfitting_threshold:
                        if best_trial is None or roc_auc > best_trial['roc_auc']:
                            best_trial = {
                                'model': model_name,
                                'trial_number': trial['Trial Number'],
                                'roc_auc': roc_auc,
                                'overfitting': overfitting,
                                'params': trial['Params']
                            }

    return best_trial

def get_best_model(best_trial_info):
    """
    Return the model instance based on the best trial information.

    This function returns a model object instantiated with the parameters from the best trial
    found in the Optuna optimization. It supports models from LightGBM, DecisionTree, XGBoost,
    RandomForest, and CatBoost.

    Args:
        best_trial_info (dict): A dictionary containing the best trial information, including
                                the model name and its parameters.

    Returns:
        model: The machine learning model instantiated with the best parameters.
    """
    model_name = best_trial_info['model']
    params = best_trial_info['params']
    
    if model_name == 'lightgbm':
        model = lgb.LGBMClassifier(**params)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(**params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_name == 'xgboost':
        model = xgb.XGBClassifier(**params)
    elif model_name == 'catboost':
        model = CatBoostClassifier(**params, verbose=0)  
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model
    

def optimize_calibration_multiclass(model, X_train, y_train):
    """
    Function to optimize the calibration of a multiclass model using Brier score.

    This function compares different calibration methods ('none', 'sigmoid', 'isotonic') for a given model 
    and training data. It computes the Brier score (a metric to assess the accuracy of predicted probabilities) 
    for each method and returns the method with the lowest score.
    
    In the whole repo, all decisions had been taken in the training set by using cross val and applied then in test set.
    Here as we previously marked a limit of 0.02 of overfitting of the trained model, we avoid cross validation (irrelevant)
    and we directly decide the best calibration method with the entire train set (cv='prefit'). 
    
    Args:
        model: The machine learning model to be calibrated.
        X_train: Training feature data.
        y_train: Training labels.

    Returns:
        tuple: The best calibration method and its corresponding Brier score.
    """
    
    # Define calibration methods to compare
    calibration_methods = ['none', 'sigmoid', 'isotonic']
    
    # Dictionary to store Brier scores and calibrated models
    brier_scores = {}
    
    # Iterate over each calibration method
    for method in calibration_methods:
        if method == 'none':
            # If no calibration, just use the original model
            calibrated_model = model
        else:
            # Apply CalibratedClassifierCV with chosen method
            calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit')
            calibrated_model.fit(X_train, y_train)
        
        # Get predicted probabilities for each class
        y_proba = calibrated_model.predict_proba(X_train)
        
        # Calculate multiclass Brier score
        brier_score = 0
        for class_idx in range(y_proba.shape[1]):  # Loop over each class
            # One-vs-rest Brier score for each class
            brier_score += brier_score_loss((y_train == class_idx).astype(int), y_proba[:, class_idx])
        
        # Average Brier score over all classes
        brier_score /= y_proba.shape[1]
        brier_scores[method] = brier_score
    
    # Get the best method based on the lowest Brier score
    best_method = min(brier_scores, key=brier_scores.get)
    return best_method, brier_scores[best_method]