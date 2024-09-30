import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class DataCleanning(BaseEstimator, TransformerMixin):
    """
    A custom transformer class for data cleaning and preprocessing.
    
    This class inherits from sklearn's BaseEstimator and TransformerMixin,
    allowing it to be used in sklearn Pipelines.

    """
      
    def __init__(self, object_to_float, object_to_int, time_to_numeric_years, months_to_numeric, multi_label_binarizer_features, nan_threshold, ids_threshold, unique_threshold):          
        """
        Initialize the DataCleanning class with user-specified parameters for various transformations.

        Args:
            object_to_float (list): Object columns to convert to float.
            object_to_int (list): Object columns to convert to int.
            time_to_numeric_years (list): Columns with time values to convert to years.
            months_to_numeric (list): Columns with month names to convert to numbers.
            multi_label_binarizer_features (list): Features to be encoded using MultiLabelBinarizer.
            nan_threshold (float): Threshold for dropping features with too many NaN values.
            ids_threshold (float): Threshold to drop columns with mostly unique values (e.g., IDs).
            unique_threshold (float): Threshold to drop columns with identical values.
        """
        
        self.object_to_float = object_to_float
        self.object_to_int = object_to_int
        self.time_to_numeric_years = time_to_numeric_years
        self.months_to_numeric = months_to_numeric
        self.multi_label_binarizer_features = multi_label_binarizer_features
        self.nan_threshold = nan_threshold
        self.ids_threshold = ids_threshold
        self.unique_threshold = unique_threshold

    
    def fit(self, X, y=None):
        """
        Identify features to transform or drop based on the criteria defined during initialization.

        Args:
            X (pd.DataFrame): Input dataset for fitting.
            y (None): Not used. Included for compatibility with scikit-learn.

        Returns:
            self: Returns the transformer object.
        """

        # Identify categorical features 
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Identify real categorical features (excluding those to be converted to float/int).
        self.real_categorical_features = [col for col in self.categorical_features if col not in self.object_to_float + self.object_to_int]

        # Initialize MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        
        # Identify and store features with many Nan values
        self.features_with_nan = self.features_with_many_nans(X, self.nan_threshold)
            
        # Identify and store categorical features that are ID-like (mostly unique values).
        self.features_ids = self.categorical_id_features(X, self.real_categorical_features, self.ids_threshold)

        # Remove categorical features where all values are the same 
        self.identical_features = self.same_value_features(X, self.real_categorical_features, self.unique_threshold)

        # Prepare data and fit MultiLabelBinarizer 
        for feature in self.multi_label_binarizer_features:
            X = self.prepare_multilabel_data(X.copy(), feature)
            self.mlb.fit(X[feature])
            self.known_labels = set(self.mlb.classes_)
 
        return self

    def transform(self, X, y=None):
        """
        Apply transformations based on what we found in the fit, to clean and preprocess the dataset.

        Args:
            X (pd.DataFrame): Input dataset for transformation.
            y (None): Not used. Included for compatibility with scikit-learn.

        Returns:
            pd.DataFrame: Transformed dataset.
        """

        X_transformed = X.copy()

        # Convert object columns to float
        for feature in self.object_to_float:
            X_transformed[feature] = self.convert_object_to_float(X_transformed[feature])
        
        # Convert object columns to int
        for feature in self.object_to_int:
            X_transformed[feature] = self.convert_object_to_int(X_transformed[feature])

        # Convert time columns to numeric values in years
        for feature in self.time_to_numeric_years:
            X_transformed[feature] = X_transformed[feature].apply(self.convert_to_years)
            
        # Convert month names to numeric values
        for feature in self.months_to_numeric:
            X_transformed[feature] = X_transformed[feature].apply(self.month_to_numeric)

        # Drop features with many NaN values
        X_transformed = X_transformed.drop(columns=self.features_with_nan)

        # Drop ID-like categorical features
        X_transformed = X_transformed.drop(columns=self.features_ids)

        # Drop categorical features where mostly all values are identical.
        X_transformed = X_transformed.drop(columns=self.identical_features)

        # Apply MultiLabelBinarizer encoding
        for feature in self.multi_label_binarizer_features:
            X_transformed = self.prepare_multilabel_data(X_transformed.copy(), feature)
            
            # Function to detect unknown labels in the test set.
            def check_unknowns(values):
                return any(item not in self.known_labels for item in values)
            
            # Create a binary feature 'Unknown' for new unseen labels.
            X_transformed['Unknown'] = X_transformed[feature].apply(check_unknowns).astype(int)

            # Transform the data using MultiLabelBinarizer
            encoded = self.mlb.transform(X_transformed[feature])
            encoded_df = pd.DataFrame(encoded, columns=self.mlb.classes_, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

        # Drop the original non-encoded MultiLabelBinarizer features.
        X_transformed.drop(columns=self.multi_label_binarizer_features, inplace=True)
        
        # Rename 'nan' column created in MLB to 'Missing_loan'
        X_transformed = self.rename_nan_column(X_transformed.copy())
         
        return X_transformed

    @staticmethod
    def convert_object_to_float(feature):
        """
        Convert object-type feature to float by cleaning underscores and coercing to numeric.

        Args:
            feature (pd.Series): Column with object-type data to be converted.

        Returns:
            pd.Series: Series converted to float, with non-convertible values set to NaN.
        """
        cleaned_series = feature.str.rstrip('_') # Clean the '_'
        feature = pd.to_numeric(cleaned_series, errors='coerce')
        return feature

    @staticmethod
    def convert_object_to_int(feature):
        """
        Convert object-type feature to integer by cleaning underscores and coercing to integer.

        Args:
            feature (pd.Series): Column with object-type data to be converted.

        Returns:
            pd.Series: Series converted to integer, with non-convertible values set to NaN.
        """
        cleaned_series = feature.str.rstrip('_') # Clean the '_'
        feature = pd.to_numeric(cleaned_series, errors='coerce', downcast='integer')
        return feature
    
    @staticmethod
    def convert_to_years(feature):
        """
        Convert a time feature to a numeric value representing years, assuming a format like 'X years Y months'.

        Args:
            feature (str or other): Feature containing time data as a string.

        Returns:
            float: Total years (including months converted to fractional years) or NaN if the conversion fails.
        """
        if isinstance(feature, str):
            parts = feature.split(' ')
            years = int(parts[0])
            months = int(parts[3])
            total_years = years + (months / 12)
            return total_years
        else:
            return np.nan  
    
    @staticmethod
    def month_to_numeric(feature):
        """
        Convert a month name to its corresponding numeric value (e.g., January -> 1).

        Args:
            feature (str or other): Feature containing a month name.

        Returns:
            int or NaN: Numeric representation of the month or NaN if the conversion fails.
        """
        if isinstance(feature, str):
            try:
                return pd.to_datetime(feature, format='%B').month
            except ValueError:
                return np.nan  
        else:
            return np.nan  

    @staticmethod
    def features_with_many_nans(X, nan_threshold):
        """
        Identify features (columns) with a proportion of NaN values greater than or equal to the specified threshold.

        Args:
            X (pd.DataFrame): Input dataset.
            nan_threshold (float): Threshold for the proportion of NaN values to remove a feature.

        Returns:
            list: List of features (column names) to remove exceeding the Nan values proportion.
        """
        nan_fraction = X.isnull().mean()
        features_to_remove = list(nan_fraction[nan_fraction >= nan_threshold].index)
        return features_to_remove

    @staticmethod
    def categorical_id_features(X, real_categorical_features, ids_threshold):
        """
        Identify categorical features that behave like IDs (mostly unique values) based on a threshold.

        Args:
            X (pd.DataFrame): Input dataset.
            real_categorical_features (list): List of categorical features to evaluate.
            ids_threshold (float): Proportion threshold above which a feature is considered ID-like.

        Returns:
            list: List of ID-like features to remove.
        """
        unique_occ = X[real_categorical_features].nunique()
        total_occ = X.shape[0]
        rate_unique_occ = unique_occ / total_occ
        features_to_remove = list(rate_unique_occ[rate_unique_occ >= ids_threshold].index)
        return features_to_remove
        
    @staticmethod
    def same_value_features(X, real_categorical_features, unique_threshold):
        """
        Identify categorical features where all values are the same or there are very few unique values.

        Args:
            X (pd.DataFrame): Input dataset.
            real_categorical_features (list): List of categorical features to evaluate.
            unique_threshold (float): Threshold for the maximum number of unique values to consider for removal.

        Returns:
            list: List of features to remove where all values are identical or have too few unique values.
        """
        unique_occ = X[real_categorical_features].nunique()
        features_to_remove = list(unique_occ[unique_occ <= unique_threshold].index)
        return features_to_remove
    
    @staticmethod
    def prepare_multilabel_data(X, feature):
        """
        Preprocess a multi-label categorical feature by converting it into a list of labels.
        Handles special cases like ' and ' delimiters and ensures that all values are lists.

        Args:
            X (pd.DataFrame): Input dataset.
            feature (str): The name of the multi-label feature to process.

        Returns:
            pd.DataFrame: Dataset with the processed feature where each value is a list of labels.
        """
        X[feature] = X[feature].astype(str)  # Ensure all values are strings
        X[feature] = X[feature].str.replace(' and ', ',', regex=False)  # Replace ' and ' with ',' to standardize the delimiter
        X[feature] = X[feature].apply(lambda x: [item.strip() for item in x.split(',') if item.strip()])  # Split the categories and strip spaces
        return X
    
    @staticmethod
    def rename_nan_column(X):
        """
        Rename any column named 'nan' to 'Missing_loan', which may be created during MultiLabelBinarizer processing.

        Args:
            X (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: Dataset with the renamed column.
        """
        for col in X.columns:
            if col.strip() == 'nan':
                X.rename(columns={col: 'Missing_loan'}, inplace=True)
        return X
        
    