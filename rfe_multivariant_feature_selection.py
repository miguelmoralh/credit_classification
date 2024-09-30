import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomRFECV(BaseEstimator, TransformerMixin):
    """
    Custom Recursive Feature Elimination with Cross-Validation (RFECV) based on a specified model and evaluation metric.
    
    This class performs recursive feature elimination, removing one feature at a time and checking if the model performance 
    (according to a specified scorer) drops significantly. If the performance does not decrease beyond a threshold, the feature 
    is considered redundant and removed. The process stops when removing features decrease more than the threshold the model performance.
    """
    def __init__(self, model, scorer, metric_direction, loss_threshold, cv):
        """
        Initializes the CustomRFECV transformer with the specified parameters.
        
        Args:
            model (estimator): The machine learning model to evaluate.
            scorer (scorer): A scoring function (e.g., ROC AUC) to evaluate the model's performance.
            metric_direction (str): Direction of optimization ("maximize" or "minimize").
            loss_threshold (float): The allowed performance drop threshold when eliminating features.
            cv (cross-validation): Cross-validation splitting strategy (e.g., StratifiedKFold).
        """
        self.model = model
        self.scorer = scorer
        self.metric_direction = "maximize"
        self.loss_threshold = loss_threshold 
        self.cv = cv 
        
    def fit(self, X, y):
        """
        Fits the CustomRFECV model to the data by recursively eliminating features and checking model performance.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target variable.
        
        Returns:
            self: The fitted transformer with features to remove stored in `self.features_to_remove_`.
        """
        self.features_to_remove_ = [] 
        features = list(X.columns)  
        improvement = True # Flag to indicate whether the feature elimination process should continue.
        iter_ = 0 

        while improvement:
            # Base dataset without the features that have been removed so far
            X_base = X.drop(columns=self.features_to_remove_) 
            base_score = self._cross_val_score(X_base, y) # Compute the base score using cross-validation
            self.best_removing_score_ = 0 # Reset the best score for this iteration
            self.worst_feature = None

            # Iterate over all remaining features to check their contribution
            for feature in features:
                X_temp = X_base.drop(columns=[feature]).copy()  # Subset of X without the current feature
                score_iter = self._cross_val_score(X_temp, y)  # Calculate the score without the current feature
                
                # Check if the score decrease is within the allowed threshold
                if base_score - score_iter <= self.loss_threshold:
                    # If maximizing the metric, check if the new score is higher than the current best
                    if (self.metric_direction == "maximize") and (score_iter > self.best_removing_score_):
                        self.best_removing_score_ = score_iter
                        self.worst_feature = feature
                    # If minimizing the metric, check if the new score is lower than the current best
                    if (self.metric_direction == "minimze") and (score_iter < self.best_removing_score_):
                        self.best_removing_score_ = score_iter
                        self.worst_feature = feature
            
            iter_ += 1
            
            # If no feature was identified for removal, stop the process
            if self.worst_feature == None:
                improvement = False
            else:
                # If a worst feature was found, remove it and continue
                self.features_to_remove_.append(self.worst_feature)
                features.remove(self.worst_feature)
                print(f"Iter: {iter_} feature removed: {self.worst_feature}, base score {base_score}, best score {self.best_removing_score_ }")
        return self

    def transform(self, X):
        """
        Transforms the dataset by removing the features identified during the fit process.
        
        Args:
            X (pd.DataFrame): The feature matrix.
        
        Returns:
            pd.DataFrame: The transformed dataset with features removed.
        """
        # Remove the features identified for elimination in the fit method
        X_selected = X.drop(columns=self.features_to_remove_)
        return X_selected

    def fit_transform(self, X, y):
        """
        Fits the CustomRFECV model and transforms the dataset by removing irrelevant features.
        
        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series or np.array): The target variable.
        
        Returns:
            pd.DataFrame: The transformed dataset after feature selection.
        """
        # Fit the model and then transform the data by removing irrelevant features.
        self.fit(X, y)
        return self.transform(X)

    def _cross_val_score(self, X, y):
        """
        Calculates the cross-validated score for a given set of features using the specified model and scorer.
        
        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series or np.array): The target variable.
        
        Returns:
            float: The mean score across all cross-validation folds.
        """
        scores = [] # List to store the scores from each fold

        # Perform cross-validation
        for train_index, val_index in self.cv.split(X, y):
            # Split the data into training and validation 
            X_train, X_val = X.iloc[train_index], X.iloc[val_index] 
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Train the model on the training data and evaluate on the test data
            self.model.fit(X_train.copy(), y_train) 
            y_pred = self.model.predict_proba(X_val.copy())  # Predict probabilities for all classes (Multiclass classification)

            # Compute the ROC AUC score for each class and average them
            roc_auc = self.scorer._score_func(y_val, y_pred, **self.scorer._kwargs)
            scores.append(roc_auc)  # Store the ROC AUC score for this fold

        return np.mean(scores)  # Return the mean score across all folds

