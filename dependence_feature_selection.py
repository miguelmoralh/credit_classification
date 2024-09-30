import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature selection based on Normalized Mutual Information (NMI).
    
    The transformer calculates the NMI between each feature (discrete or discretized) and the target variable, 
    as well as the NMI between pairs of features. Features that are highly correlated (based on NMI) are 
    removed, prioritizing the feature with lower NMI to the target.
    """
    def __init__(self, continuous_features, target, num_bins=5, threshold=0.5, average_method='min'):
        """
        Initialize the FeatureSelection transformer with the given parameters.

        Args:
            continuous_features (list): List of continuous features to be discretized.
            target (pd.Series): The target variable to compute the mutual information score with.
            num_bins (int, optional): Number of bins for discretizing continuous features. Defaults to 5.
            threshold (float, optional): Mutual Information threshold for dropping dependent features. Defaults to 0.5.
            average_method (str, optional): Method for averaging in NMI calculation. Defaults to 'min'.
        """
        self.continuous_features = continuous_features
        self.target = target
        self.num_bins = num_bins
        self.threshold = threshold
        self.average_method = average_method
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the input data by calculating Normalized Mutual Information (NMI) between
        features and the target, as well as between pairs of features. Identify features to drop based
        on the NMI values.

        Args:
            X (pd.DataFrame): The input dataset with features to evaluate.
            y (pd.Series, optional): Not used in this implementation.

        Returns:
            self: The fitted transformer with identified features to drop stored in `self.features_to_drop`.
        """
        
        # Identify original discrete features
        self.discrete_features = [col for col in X.columns if col not in self.continuous_features]

        # Discretize (binning) continuous features and create a binned dataset
        X_binned = X.copy()
        for col in self.continuous_features:
            X_binned[col] = pd.cut(X[col], bins=self.num_bins, labels=False)
        
        # Calculate NMI between each feature (discrete or discretized) and the target
        self.nmi_with_target = [
            normalized_mutual_info_score(X_binned[col], self.target, average_method=self.average_method) for col in X_binned.columns
        ]
        
        # Store the NMI values in a Pandas Series, sorted in descending order
        self.nmi_series_with_target = pd.Series(self.nmi_with_target, index=X_binned.columns).sort_values(ascending=False)
        
        # Create an empty matrix to store NMI values between pairs of features
        nmi_matrix = pd.DataFrame(index=X_binned.columns, columns=X_binned.columns)
        
        # Calculate NMI for each unique pair of features (upper triangular matrix)
        for col1, col2 in combinations(X_binned.columns, 2): 
            nmi_value = normalized_mutual_info_score(X_binned[col1], X_binned[col2], average_method=self.average_method)
            nmi_matrix.loc[col1, col2] = nmi_value
    
        # Identify pairs of features with NMI above the threshold and mark one feature for removal
        self.features_to_drop = set()
        for col1, col2 in combinations(X_binned.columns, 2):
            if nmi_matrix.loc[col1, col2] > self.threshold:
                
                # Drop the feature with the lower NMI with the target
                if self.nmi_series_with_target[col1] < self.nmi_series_with_target[col2]:
                    self.features_to_drop.add(col1)
                elif self.nmi_series_with_target[col1] > self.nmi_series_with_target[col2]:
                    self.features_to_drop.add(col2)
                else:
                    self.features_to_drop.add(np.random.choice([col1, col2]))
        
        self.features_to_drop = list(self.features_to_drop)
        print('Dependence Dropped features: ', self.features_to_drop)
        
        return self

    def transform(self, X, y=None):
        """
        Transform the dataset by dropping the features identified during fitting as having high mutual information.

        Args:
            X (pd.DataFrame): The input dataset to transform.
            y (pd.Series, optional): Not used in this implementation.

        Returns:
            pd.DataFrame: The transformed dataset with high-dependence features dropped.
        """
        # Drop features identified in the fit method as having high mutual information
        X = X.drop(columns=self.features_to_drop, errors='ignore')
        return X

