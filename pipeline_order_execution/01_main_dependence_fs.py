from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, CONTINUOUS_FEATURES, TARGET_MAPPING
)
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
from dependence_feature_selection import FeatureSelection
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load the dataset into a pandas DataFrame
df = pd.read_csv('data/train.csv')

# Separate the target from the dataframe
x = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Encode the target variable 'y' using the provided mapping
y_encoded = y.map(TARGET_MAPPING)

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=4)

### DATASET CLEANNING

data_cleaner = DataCleanning(
    object_to_float=OBJECT_TO_FLOAT,                      
    object_to_int=OBJECT_TO_INT,
    time_to_numeric_years=TIME_TO_NUMERIC_YEARS, 
    months_to_numeric=MONTHS_TO_NUMERIC,
    multi_label_binarizer_features=MULTI_LABEL_BINARIZER_FEATURES,
    nan_threshold = 0.9,
    ids_threshold = 0.12, 
    unique_threshold = 1
)

# Clean the training and test data
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

### IMPUTE MISSING VALUES

imputer = ImputeMissing()  

# Impute missing values in training and test data
X_train_imputed = imputer.fit_transform(X_train_cleaned)
X_test_imputed = imputer.transform(X_test_cleaned)

### ENCODE CATEGORICAL VARIABLES

encoder = CategoricalEncoder(
    ordinal_variables=ORDINAL_VARIABLES, 
    categorical_variables=CATEGORICAL_NON_ORDINAL_VARIABLES
)

# Encode training and test data
X_train_encoded = encoder.fit_transform(X_train_imputed)
X_test_encoded = encoder.transform(X_test_imputed)

### DEPENDENCE BIVARIANT FEATURE SELECTION

feature_selector = FeatureSelection(
    continuous_features=CONTINUOUS_FEATURES, 
    target=y_train, 
    num_bins=10, 
    threshold=0.7,
    average_method='min'
)

# Select features from the encoded training and test data
X_train_selected = feature_selector.fit_transform(X_train_encoded)
X_test_selected = feature_selector.transform(X_test_encoded)
selected_features = X_train_selected.columns.tolist()

# Define the path to save the selected features
logs_dir = 'logs/selected_features'
dependence_selected_features_file = os.path.join(logs_dir, 'dependence_selected_features.txt')

# Save the names of the selected features to a text file
with open(dependence_selected_features_file, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Selected features saved to {dependence_selected_features_file}")