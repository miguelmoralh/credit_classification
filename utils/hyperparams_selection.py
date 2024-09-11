from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb

def get_model(trial, model_type, cat_features):
    if model_type == "decision_tree":
        max_depth = trial.suggest_int("max_depth", 2, 40) 
        min_samples_split = trial.suggest_int("min_samples_split", 2, 32) # Minimum samples required to split an internal node
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32) # Minimum samples required to create a leaf
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) # Number of features considered when looking for the best split
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy']) # Function to measure the quality of the split
        splitter = trial.suggest_categorical('splitter', ['best', 'random']) # Strategy used to choose the split
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
        n_estimators = trial.suggest_int("n_estimators", 100, 1000) # Number of decision trees
        max_depth = trial.suggest_int("max_depth", 5, 30) # Maximum depth of each decision tree
        criterion = trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]) # Function to measure the quality of the split
        min_samples_split = trial.suggest_int("min_samples_split", 2, 32) # Minimum samples required to split an internal node
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 32) # Minimum samples required to create a leaf
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None]) # Number of features considered when looking for the best split
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
        n_estimators = trial.suggest_int("n_estimators", 1000, 3000) # Number of decision trees
        max_depth = trial.suggest_int("max_depth", 1, 10) # Maximum depth of each estimator
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1) # Scales the contribution of each decision tree
        subsample = trial.suggest_float('subsample', 0.1, 1.0, log=True) # Proportion of dataset to be considered when building each tree
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True) # Proportion of features to be considered for each tree 
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20) # Minimum sum of instances in a child node (every node except the root node)
        alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True) # L1 regularization weight
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
        
        '''
        The log scale is used for the learning rate because it will try more values close to 0.001, 
        as small learning rates with a large number of trees tend to be more stable.
        
        '''
    elif model_type == "catboost":
        iterations = trial.suggest_int("iterations", 1000, 3000) # Number of decision trees (estimators)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True) # Scales the contribution of each decision tree
        depth = trial.suggest_int("depth", 1, 10) # Maximum depth of each estimator
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.1, 1.0, log=True) # Proportion of features to be considered when determining the best split
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 100) # Minimum number of samples required to create a leaf
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ["Bernoulli", "Bayesian"]) # Defines the method for sampling the weights of object
        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0, 10) # Use the Bayesian bootstrap to assign random weights to objects.
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                colsample_bylevel=colsample_bylevel,
                min_data_in_leaf=min_data_in_leaf,
                bootstrap_type=bootstrap_type, # Defines the method for sampling the weights of object
                bagging_temperature=bagging_temperature,
                random_state=42,
                cat_features=cat_features,
                objective='MultiClass',  # Specify the objective for multiclass classification
                verbose=0 # Turn off logging for CatBoost
            )
            
        elif bootstrap_type == "Bernoulli":
            subsample = trial.suggest_float('subsample', 0.1, 1.0, log=True) # Proportion of dataset to be considered when building each tree
            return CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                colsample_bylevel=colsample_bylevel,
                min_data_in_leaf=min_data_in_leaf,
                bootstrap_type=bootstrap_type, # Defines the method for sampling the weights of object
                subsample=subsample,
                random_state=42,
                cat_features=cat_features,
                objective='MultiClass',  # Specify the objective for multiclass classification
                verbose=0 # Turn off logging for CatBoost
            )
    elif model_type == "lightgbm":
        n_estimators = trial.suggest_int("n_estimators", 1000, 3000) # Number of decision trees
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True) # Scales the contribution of each decision tree
        num_leaves = trial.suggest_int("num_leaves", 2, 2**10) # Maximum number of terminal nodes (leaves)
        subsample = trial.suggest_float('subsample', 0.1, 1, log=True) # Proportion of dataset to be randomly considered when training each tree
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0, log=True) # Proportion of features to be considered for each tree
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 100) # Minimum number of samples in a leaf node
        lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True) # L1 regularization
        lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True) # L2 regularization 
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            bagging_freq = 1, # Frequency at which data is sampled (1 means that is resampled before every tree) 
            subsample=subsample, 
            colsample_bytree=colsample_bytree,
            min_data_in_leaf = min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            random_state=42,
            verbosity=-1,
            objective="multiclass", # Set objective for multiclass classification
            num_class=3 # Set number of classes
        )
    
def objective(trial, model_type, X, y, imputer, encoder, scorer, scv):
    model = get_model(trial, model_type, cat_features=None)
    pipe = Pipeline([
        ('imputer', imputer),
        ('encoder', encoder),
        ('model', model)
    ])
    scores_val_list = []
    scores_overfitting_list = []
    
    # Stratified cross validation
    for train_index, val_index in scv.split(X.copy(), y.copy()):
        X_train, X_val = X.copy().iloc[train_index], X.copy().iloc[val_index]
        y_train, y_val = y.copy().iloc[train_index], y.copy().iloc[val_index]
        
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict_proba(X_train)
        y_pred_test = pipe.predict_proba(X_val)
        
        # Save score in train to calculate the overfitting 
        score_train = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)
        score_val = scorer._score_func(y_val, y_pred_test, **scorer._kwargs)
        overfitting_score = score_train - score_val
        scores_val_list.append(score_val)
        scores_overfitting_list.append(overfitting_score)
    
    mean_score = np.mean(scores_val_list)
    mean_overfitting_score = np.mean(scores_overfitting_list)
    
    return mean_score, mean_overfitting_score

# Function done to treat catboost as no need to encode categorical features (NO PIPELINE HERE AS DIFFICULTIES BETWEEN SKLEARN AND OTHER LIBRARIES)
def objective_categorical(trial, model_type, X, y, imputer, scorer, scv, cat_features): 
    model = get_model(trial, model_type, cat_features)
    scores_val_list = []
    scores_overfitting_list = []
    for train_index, val_index in scv.split(X.copy(), y.copy()):
        X_train, X_val = X.copy().iloc[train_index], X.copy().iloc[val_index]
        y_train, y_val = y.copy().iloc[train_index], y.copy().iloc[val_index]
        
        # Impute missing values
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_val)
        
        model.fit(X_train_imputed, y_train)
        y_pred_train = model.predict_proba(X_train_imputed)
        y_pred_test = model.predict_proba(X_test_imputed)
        
        # Save score in train to calculate the overfitting 
        score_train = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)
        score_val = scorer._score_func(y_val, y_pred_test, **scorer._kwargs)
        overfitting_score = score_train - score_val
        scores_val_list.append(score_val)
        scores_overfitting_list.append(overfitting_score)

    mean_score = np.mean(scores_val_list)
    mean_overfitting_score = np.mean(scores_overfitting_list)
    
    return mean_score, mean_overfitting_score