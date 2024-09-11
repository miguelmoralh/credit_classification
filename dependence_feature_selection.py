import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
from sklearn.metrics import normalized_mutual_info_score

class FeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self, continuous_features, target, num_bins=5, threshold=0.5, average_method='min'):
        self.continuous_features = continuous_features
        self.target = target
        self.num_bins = num_bins
        self.threshold = threshold
        self.average_method = average_method
        
    def fit(self, X, y=None):
        
        # Identify original discrete features
        self.discrete_features = [col for col in X.columns if col not in self.continuous_features]

        # Discretize (binning) continuous features and create a binned dataset
        X_binned = X.copy()
        for col in self.continuous_features:
            X_binned[col] = pd.cut(X[col], bins=self.num_bins, labels=False)
        
        # Calculate Normalized Mutual Info between each feature (discrete or discretized) and the target
        self.nmi_with_target = [normalized_mutual_info_score(X_binned[col], self.target, average_method=self.average_method) for col in X_binned.columns]
        self.nmi_series_with_target = pd.Series(self.nmi_with_target, index=X_binned.columns).sort_values(ascending=False)
        
        # Check MI with target
        # print("Normalized Mutual Information with target:\n", self.nmi_series_with_target)
        
        # Calculate Normalized Mutual Info between all pairs of features (only upper triangular part)
        nmi_matrix = pd.DataFrame(index=X_binned.columns, columns=X_binned.columns)
        for col1, col2 in combinations(X_binned.columns, 2):  # combinations from itertools is used to avoid pairs repetitions
            nmi_value = normalized_mutual_info_score(X_binned[col1], X_binned[col2], average_method=self.average_method)
            nmi_matrix.loc[col1, col2] = nmi_value
    
        # print("Mutual Information upper triangular matrix:\n", nmi_matrix)

        # Identify pairs of features with MI above the threshold. Pairs of features with a high corr means that both explain similar info (one should be delete it)
        self.features_to_drop = set()
        for col1, col2 in combinations(X_binned.columns, 2):
            if nmi_matrix.loc[col1, col2] > self.threshold:
                
                # Drop the feature with the lower MI with the target (the feature less related with the target)
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
        # Drop features with a MI below or equal the threshold
        X = X.drop(columns=self.features_to_drop, errors='ignore')
        return X

