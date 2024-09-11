import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomRFECV(BaseEstimator, TransformerMixin):
    def __init__(self, model, scorer, metric_direction, loss_threshold, cv):
        self.model = model
        self.scorer = scorer
        self.metric_direction = "maximize"
        self.loss_threshold = loss_threshold # Threshold of error for eliminate features according to ROC AUC score
        self.cv = cv # To compute ROC AUC for multiclass targets (one vs one)
        
    def fit(self, X, y):
        self.features_to_remove_ = []  
        features = list(X.columns)  # List of all feature names
        improvement = True 
        iter_ = 0

        while improvement:
            X_base = X.drop(columns=self.features_to_remove_) 
            base_score = self._cross_val_score(X_base, y) 
            self.best_removing_score_ = 0
            self.worst_feature = None

            for feature in features:
                X_temp = X_base.drop(columns=[feature]).copy()  # Subset of X without the current feature
                score_iter = self._cross_val_score(X_temp, y)  # Calculate the ROC AUC score without the current feature
                
                # If the score does not decrease more than the threshold, check if is the worse feature according to the score FOR REMOVAL in the iter
                if base_score - score_iter <= self.loss_threshold:
                    if (self.metric_direction == "maximize") and (score_iter > self.best_removing_score_):
                        self.best_removing_score_ = score_iter
                        self.worst_feature = feature
                    if (self.metric_direction == "minimze") and (score_iter < self.best_removing_score_):
                        self.best_removing_score_ = score_iter
                        self.worst_feature = feature
            
            iter_ += 1
            
            # Once all redundant features (the ones that do not affect the score) are eliminated, it ends
            if self.worst_feature == None:
                improvement = False
            else:
                self.features_to_remove_.append(self.worst_feature)
                features.remove(self.worst_feature)
                print(f"Iter: {iter_} feature removed: {self.worst_feature}, base score {base_score}, best score {self.best_removing_score_ }")
        return self

    def transform(self, X):
        
        # removing the features that were found to be non-informative
        X_selected = X.drop(columns=self.features_to_remove_)
        return X_selected

    def fit_transform(self, X, y):
        
        # Fits the model and then transforms the dataset
        self.fit(X, y)
        return self.transform(X)

    def _cross_val_score(self, X, y):
        # Stratified K-Folds cross-validator
        scores = []

        # Perform cross-validation
        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Split the data
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # For each fold, DO THE ENTIRE DATA PREPROCESSING and train the model
            self.model.fit(X_train.copy(), y_train) 
            y_pred = self.model.predict_proba(X_test.copy())  # Predict probabilities for all classes (Multiclass classification)

            # Compute the ROC AUC score for each class and average them
            roc_auc = self.scorer._score_func(y_test, y_pred, **self.scorer._kwargs)
            scores.append(roc_auc)  # Store the average ROC AUC score

        return np.mean(scores)  # Return the mean ROC AUC score across all folds

