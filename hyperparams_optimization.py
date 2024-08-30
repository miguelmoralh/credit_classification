import optuna
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
import numpy as np
from catboost import CatBoostClassifier
import lightgbm as lgb



def get_model(trial, model_type):
    if model_type == "random_forest":
        n_estimators = trial.suggest_int("rf_n_estimators", 10, 200)
        max_depth = trial.suggest_int("rf_max_depth", 2, 15)
        min_samples_leaf = trial.suggest_float("rf_min_samples_leaf", 0.001, 0.1)
        criterion = trial.suggest_categorical("rf_criterion", ["gini", "entropy", "log_loss"])
        max_features = trial.suggest_float("rf_max_features", 0.4, 1.0)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            max_features=max_features,
            random_state=42
        )
    elif model_type == "xgboost":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 1000)
        max_depth = trial.suggest_int("xgb_max_depth", 2, 15)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
        subsample = trial.suggest_float('xgb_subsample', 0.4, 1.0)
        colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.4, 1.0)
        gamma = trial.suggest_float('xgb_gamma', 0, 5)
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma
            
        )
    elif model_type == "adaboost":
        n_estimators = trial.suggest_int("ab_n_estimators", 50, 500)
        learning_rate = trial.suggest_float("ab_learning_rate", 0.01, 1.0)
        return AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    elif model_type == "decision_tree":
        max_depth = trial.suggest_int("dt_max_depth", 2, 32)
        min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("dt_min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical('dt_max_features', ['sqrt', 'log2', None])
        criterion = trial.suggest_categorical('dt_criterion', ['gini', 'entropy'])
        splitter = trial.suggest_categorical('dt_splitter', ['best', 'random'])
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            splitter=splitter,
            random_state=42
        )
    elif model_type == "catboost":
        depth = trial.suggest_int("cb_depth", 4, 15)
        learning_rate = trial.suggest_float("cb_learning_rate", 0.01, 0.3)
        iterations = trial.suggest_int("cb_iterations", 100, 1000)
        l2_leaf_reg = trial.suggest_float('cb_l2_leaf_reg', 1, 10)
        bagging_temperature = trial.suggest_float('cb_bagging_temperature', 0, 1)
        return CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_state=42
        )
    elif model_type == "lightgbm":
        num_leaves = trial.suggest_int("lg_num_leaves", 10, 200)
        learning_rate = trial.suggest_float("lg_learning_rate", 0.01, 0.3)
        n_estimators = trial.suggest_int("lg_n_estimators", 100, 1000)
        min_child_samples = trial.suggest_float('lg_min_child_samples', 10, 100)
        subsample = trial.suggest_float('lg_subsample', 0.5, 1)
        colsample_bytree = trial.suggest_float('lg_colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_float('lg_reg_alpha', 0, 10)
        reg_lambda = trial.suggest_float('lg_reg_lambda', 0, 10)
        return lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42
        )
    
      
def objective(trial, model_type, X, y, imputer, encoder, scorer, scv):
    model = get_model(trial, model_type)
    pipe = Pipeline([
        ('imputer', imputer),
        ('encoder', encoder),
        ('model', model)
    ])
    
    scores_val_list = []
    scores_overfitting_list = []
    
    for train_index, test_index in scv.split(X.copy(), y.copy()):
        X_train, X_test = X.copy().iloc[train_index], X.copy().iloc[test_index]
        y_train, y_test = y.copy().iloc[train_index], y.copy().iloc[test_index]
        
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict_proba(X_train)
        y_pred_test = pipe.predict_proba(X_test)
        
        # Save score in train to calculate the overfitting and optimize the maximization of the score in val and the minimization of the overfitting
        score_train = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)
        score_val = scorer._score_func(y_test, y_pred_test, **scorer._kwargs)
        overfitting_score = score_train - score_val
        scores_val_list.append(score_val)
        scores_overfitting_list.append(overfitting_score)
    
    mean_score = np.mean(scores_val_list)
    mean_overfitting_score = np.mean(scores_overfitting_list)
    
    return mean_score, mean_overfitting_score
