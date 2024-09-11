import json
import optuna.visualization as vis
import os
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

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
