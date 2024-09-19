import json
import optuna.visualization as vis
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def match_features(features_list, df):
    """
    Return a list of the common features between a dataframe and a list of columns taking 
    the differences in the names as ones appear encoded and other not.

    """
    matched_features = []
    for feature in features_list:
        if feature in df.columns:
            matched_features.append(feature)
        elif feature.endswith('_encoded'):
            base_feature = feature[:-8]  # Remove '_encoded' suffix
            if base_feature in df.columns:
                matched_features.append(base_feature)
    return matched_features

def save_study_trials_to_json(study, model_name, output_dir="logs"):
    """
    Save the trials of an Optuna study to a JSON file.
    
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
    
def save_pareto_front_plot(study, model_name, output_dir="logs"):
    """
    Plot and save the Pareto front for an Optuna study.
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
    Function to find the best trial over all models from the json optuna optimization studies. 
    
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
                    
                    # Si el overfitting es menor que el umbral, evaluamos si es el mejor ROC AUC hasta ahora
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
    This function returns the model instance based on the model name in the best_trial_info.
    It handles LightGBM, DecisionTree, XGBoost, RandomForest, and CatBoost.
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
        model = CatBoostClassifier(**params, verbose=0)  # verbose=0 to suppress CatBoost training logs
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    return model

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
    
'''
Remember that the decisions ARE TAKEN WITH TRAIN DATA BY USING VALIDATION, AND THEN CHANGES ARE APPLIED TO TEST.
That is why we use only training data in the function.

The Brier score measures the mean squared difference between the predicted probabilities 
and the actual outcomes. Lower Brier scores indicate better calibration.
'''

# Function to evaluate calibration methods and return the best one based on Brier score
def optimize_calibration_multiclass(model, X_train, y_train):
    
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



