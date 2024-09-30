from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ImputeMissing(BaseEstimator, TransformerMixin):
    """
    Custom transformer for imputing missing values in a dataset.

    This class imputes missing values in numerical features using the median 
    and in categorical features using the string 'Missing'.
    
    Inherits from sklearn's BaseEstimator and TransformerMixin for compatibility 
    with scikit-learn's pipeline.
    """
    def __init__(self):
        """
        Initialize the ImputeMissing transformer.
        
        No parameters are needed for initialization, as all logic is defined
        during the fit and transform methods.
        """
        pass
        
    def fit(self, X, y=None):
        """
        Fit the imputer by identifying features with missing values and 
        calculating the median for numerical features.
        
        Args:
            X (pd.DataFrame): Input dataset to fit the imputer on.
            y (pd.Series, optional): Target variable, not used here.
        
        Returns:
            self: Fitted transformer with calculated median values for numerical features.
        """
        
        # Identify numerical and categorical features with missing values
        self.numerical_features = [col for col in X.select_dtypes(include=[np.number]).columns if X[col].isnull().any()]
        self.categorical_features = [col for col in X.select_dtypes(include=[object]).columns if X[col].isnull().any()]

        # Calculate the median for each numerical feature with Nan values to be used for imputation
        self.median_values = {feature: X[feature].dropna().median() for feature in self.numerical_features}

        return self

    def transform(self, X, y=None):
        """
        Transform the dataset by imputing missing values in both numerical 
        and categorical features.
        
        Args:
            X (pd.DataFrame): Input dataset to transform.
            y (pd.Series, optional): Target variable, not used here.
        
        Returns:
            pd.DataFrame: Transformed dataset with missing values imputed.
        """

        X_impute = X.copy()

        # Impute missing values in numerical features using the precomputed medians
        for feature in self.numerical_features:
            X_impute[feature].fillna(self.median_values[feature], inplace=True)
            
       # Impute missing values in categorical features using the string 'Missing'.
        for feature in self.categorical_features:
            X_impute[feature].fillna('Missing', inplace=True)
    
        return X_impute