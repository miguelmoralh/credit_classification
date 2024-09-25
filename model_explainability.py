import shap
import pandas as pd
from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES
)
from utils.utils import match_features, find_best_trial, get_best_model, optimize_calibration_multiclass
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import os
# Load dataframe
df = pd.read_csv('data/train.csv')

# Separate the target from the dataframe
x = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Define the mapping for the target variable
target_mapping = {
    "Poor": 0,
    "Standard": 1,
    "Good": 2
}

# Encode the target variable 'y' using the mapping
y_encoded = y.map(target_mapping)

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
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

### APPLY FEATURE SELECTION FROM SAVED IN logs txts

# Get the path of the stored selected features in 01_main_dependence_fs.py and 02_main_rfe_fs.py
file_path = "logs\\rfe_selected_features.txt"

# Open the file and read the lines
with open(file_path, 'r') as file:
    features_list = [line.strip() for line in file.readlines()]

# Filter the common columns between features_list and X_train_cleaned (handling encoding renaming columns)
original_features = match_features(features_list, X_train_cleaned)

# Filter the dataset with dependence selected features
X_train_filtered = X_train_cleaned[original_features]
X_test_filtered = X_test_cleaned[original_features]

### IMPUTE MISSING VALUES
imputer = ImputeMissing()  
X_train_imputed = imputer.fit_transform(X_train_filtered)
X_test_imputed = imputer.transform(X_test_filtered)

### ENCODE CATEGORICAL VARIABLES

# If NO categorical features in the dataframe, encoder deals with that and does nothing
encoder = CategoricalEncoder(
    ordinal_variables=ORDINAL_VARIABLES, 
    categorical_variables=CATEGORICAL_NON_ORDINAL_VARIABLES
)
X_train_encoded = encoder.fit_transform(X_train_imputed)
X_test_encoded = encoder.transform(X_test_imputed)

# Load the saved calibrated model (to ensure it's properly saved)
loaded_model = joblib.load('logs/calibrated_trained_model.pkl')

'''
I am going to use SHAP ((SHapley Additive exPlanations) to explain the trained and calibrated machine learning model.
SHAP is based on game theory and provides consistent global or local explanations for the predictions of models.

SHAP measures the individual contribution of each feature to a model's prediction. It uses Shapley values, 
a concept from game theory, which assigns each feature a fair contribution to the model’s prediction.

'''
### MODEL EXPLAINABILITY USING SHAP

# Extract the base model from CalibratedClassifierCV
base_model = loaded_model.calibrated_classifiers_[0].estimator  # This accesses the original RandomForestClassifier

# Create the SHAP explainer for tree-based models (Random Forest)
explainer = shap.TreeExplainer(base_model)

# Calculate SHAP values for all instances
shap_values = explainer.shap_values(X_test_encoded)

# For Class 0
shap.summary_plot(shap_values[:, :, 0], X_test_encoded, show=False)
plt.title("Class 0 Summary")
plt.savefig('logs/shap_summary_class_0.png', bbox_inches='tight')
plt.close()

# For Class 1
shap.summary_plot(shap_values[:, :, 1], X_test_encoded, show=False)
plt.title("Class 1 Summary")
plt.savefig('logs/shap_summary_class_1.png', bbox_inches='tight')
plt.close()

# For Class 2
shap.summary_plot(shap_values[:, :, 2], X_test_encoded, show=False)
plt.title("Class 2 Summary")
plt.savefig('logs/shap_summary_class_2.png', bbox_inches='tight')
plt.close()
