from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ImputeMissing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        
        # Identify numerical and categorical features with missing values
        self.numerical_features = [col for col in X.select_dtypes(include=[np.number]).columns if X[col].isnull().any()]
        self.categorical_features = [col for col in X.select_dtypes(include=[object]).columns if X[col].isnull().any()]

        # Calculate the median of numerical features to be used for imputation
        self.median_values = {feature: X[feature].dropna().median() for feature in self.numerical_features}

        return self

    def transform(self, X, y=None):

        X_impute = X.copy()

        # Impute missings in numerical features with the median
        for feature in self.numerical_features:
            X_impute[feature].fillna(self.median_values[feature], inplace=True)
            
        # In case we had categorical features with missings
        # Impute categorical features with 'Missing'
        for feature in self.categorical_features:
            X_impute[feature].fillna('Missing', inplace=True)
    
        # Rename any variation of 'nan' with spaces to 'missing_loan'
        for col in X_impute.columns:
            if col.strip() == 'nan':
                X_impute.rename(columns={col: 'missing_loan'}, inplace=True)

        return X_impute