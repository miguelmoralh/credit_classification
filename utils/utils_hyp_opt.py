import optuna.visualization as vis
import json
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

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