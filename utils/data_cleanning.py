import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class DataCleanning(BaseEstimator, TransformerMixin):
    def __init__(self, object_to_float, object_to_int, time_to_numeric_years, months_to_numeric, multi_label_binarizer_features, nan_threshold, ids_threshold, unique_threshold):
        self.object_to_float = object_to_float
        self.object_to_int = object_to_int
        self.time_to_numeric_years = time_to_numeric_years
        self.months_to_numeric = months_to_numeric
        self.multi_label_binarizer_features = multi_label_binarizer_features
        self.nan_threshold = nan_threshold
        self.ids_threshold = ids_threshold
        self.unique_threshold = unique_threshold

    
    def fit(self, X, y=None):

        # Identify the filtered and real categorical features to pass them through some functions
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        self.real_categorical_features = [col for col in self.categorical_features if col not in self.object_to_float + self.object_to_int]

        # Initialize MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer()
        
        # Remove features with more than nan_threshold NaN values
        self.features_with_nan = self.features_with_many_nans(X, self.nan_threshold)
        print(self.features_with_nan, 'nan')
            
        # Remove categorical features that are IDs or almost IDs (unique values, based on threshold)
        self.features_ids = self.categorical_id_features(X, self.real_categorical_features, self.ids_threshold)
        print(self.features_ids, 'id')

        # Remove categorical features where all values are the same 
        self.identical_features = self.same_value_features(X, self.real_categorical_features, self.unique_threshold)
        print(self.identical_features, 'identical')

        # Prepare data and fit MultiLabelBinarizer 
        for feature in self.multi_label_binarizer_features:
            X = self.prepare_multilabel_data(X.copy(), feature)
            self.mlb.fit(X[feature])
            self.known_labels = set(self.mlb.classes_)
 
        
        return self

    def transform(self, X, y=None):

        X_transformed = X.copy()

        # Clean and transform object features with numerical info to numerical features (float and int)
        for feature in self.object_to_float:
            X_transformed[feature] = self.convert_object_to_float(X_transformed[feature])

        for feature in self.object_to_int:
            X_transformed[feature] = self.convert_object_to_int(X_transformed[feature])

        # Transform time features to numeric in YEARS
        for feature in self.time_to_numeric_years:
            X_transformed[feature] = X_transformed[feature].apply(self.convert_to_years)
            
        # Transform month names to numeric using pd.to_datetime
        for feature in self.months_to_numeric:
            X_transformed[feature] = X_transformed[feature].apply(self.month_to_numeric)

        # Remove features with more than nan_threshold NaN values
        X_transformed = X_transformed.drop(columns=self.features_with_nan)

        # Remove categorical features that are IDs or almost IDs (unique values, based on threshold)
        X_transformed = X_transformed.drop(columns=self.features_ids)

        # Remove categorical features where all values are the same 
        X_transformed = X_transformed.drop(columns=self.identical_features)

        # Prepare data for MultiLabelBinarizer encoding for special variables
        for feature in self.multi_label_binarizer_features:
            X_transformed = self.prepare_multilabel_data(X_transformed.copy(), feature)
            
            # Check for unknown labels and add 'Unknown' column
            def check_unknowns(values):
                return any(item not in self.known_labels for item in values)
            
            X_transformed['Unknown'] = X_transformed[feature].apply(check_unknowns).astype(int)

            # Transform the data using MultiLabelBinarizer
            encoded = self.mlb.transform(X_transformed[feature])
            encoded_df = pd.DataFrame(encoded, columns=self.mlb.classes_, index=X_transformed.index)
            X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

        # Finally eliminate the original, not encoded 'Type_of_Loan'
        X_transformed.drop(columns=self.multi_label_binarizer_features, inplace=True)
        
        
        return X_transformed

    @staticmethod
    def convert_object_to_float(feature):
        cleaned_series = feature.str.rstrip('_') # Clean the '_'
        feature = pd.to_numeric(cleaned_series, errors='coerce')
        return feature

    @staticmethod
    def convert_object_to_int(feature):
        cleaned_series = feature.str.rstrip('_') # Clean the '_'
        feature = pd.to_numeric(cleaned_series, errors='coerce', downcast='integer')
        return feature
    
    @staticmethod
    def convert_to_years(feature):
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
        if isinstance(feature, str):
            try:
                return pd.to_datetime(feature, format='%B').month
            except ValueError:
                return np.nan  
        else:
            return np.nan  

    @staticmethod
    def features_with_many_nans(X, nan_threshold):
        nan_fraction = X.isnull().mean()
        features_to_remove = list(nan_fraction[nan_fraction >= nan_threshold].index)
        return features_to_remove

    @staticmethod
    def categorical_id_features(X, real_categorical_features, ids_threshold):
        unique_occ = X[real_categorical_features].nunique()
        total_occ = X.shape[0]
        rate_unique_occ = unique_occ / total_occ
        features_to_remove = list(rate_unique_occ[rate_unique_occ >= ids_threshold].index)
        return features_to_remove
        
    @staticmethod
    def same_value_features(X, real_categorical_features, unique_threshold):
        unique_occ = X[real_categorical_features].nunique()
        features_to_remove = list(unique_occ[unique_occ <= unique_threshold].index)
        return features_to_remove
        
    @staticmethod
    def prepare_multilabel_data(df, feature):
        df[feature] = df[feature].astype(str)  # Ensure all values are strings
        df[feature] = df[feature].str.replace(' and ', ',', regex=False)  # Replace ' and ' with ',' to standardize the delimiter
        df[feature] = df[feature].apply(lambda x: [item.strip() for item in x.split(',') if item.strip()])  # Split the categories and strip spaces
        return df