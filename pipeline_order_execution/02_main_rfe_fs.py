'''
Explanation: Before starting training and evaluating the final model I have to do 3 steps, the features selection, 
the selection of the model and the optimization of hyperparameters of the model.The best solution would be to run 
these 3 steps at the same time one after the other, but it is computationally too expensive. So, the way I select the 
features with RFE is by choosing a model that we know it will give a standard and good performance (Random Forest) and 
choosing a common and standard hyperparameeters for the model. Once I have the model and the hyperparams we can train/run
the model to see how the score (ROC AUC) is affected with the different combinations of features. Then, based on the score and
a threshold we keep or eliminate features.
'''

from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, PARAMS_RF_RFE
)
import os
from utils.utils import match_features
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
from rfe_multivariant_feature_selection import CustomRFECV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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

# Get the path of the stored selected features in 01_main_dependence_fs.py
file_path = "logs\dependence_selected_features.txt"

# Open the file and read the lines
with open(file_path, 'r') as file:
    features_list = [line.strip() for line in file.readlines()]

# Filter the common columns between features_list and X_train_cleaned (handling encoding renaming columns)
original_features = match_features(features_list, X_train_cleaned)

# Filter the dataset with dependence selected features
X_train_filtered = X_train_cleaned[original_features]
X_test_filtered = X_test_cleaned[original_features]

# Initialize imputer and encoder
imputer = ImputeMissing()  
encoder = CategoricalEncoder(
    ordinal_variables=ORDINAL_VARIABLES, 
    categorical_variables=CATEGORICAL_NON_ORDINAL_VARIABLES
)

### MULTIVARIANT FEATURE SELECTION : RECURSIVE FEATURE ELIMINATION (RFE)

# Use Random Forest as model 
rf_clf = RandomForestClassifier(**PARAMS_RF_RFE)  # ** to initialize model from dict 
pipe_rf = Pipeline(  # Pipline to apply the data pre processing to each fold in the cross val
    [
        ('imputer', imputer),
        ('encoder', encoder),
        ("model", rf_clf)
    ]
)

# Create a custom scorer for multiclass ROC AUC
scorer = make_scorer(roc_auc_score, multi_class='ovo', needs_proba=True)

rfe = CustomRFECV(
    model=pipe_rf,
    scorer=scorer,
    metric_direction="maximize",
    loss_threshold=0.001,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

X_train_selected_rfe = rfe.fit_transform(X_train_filtered.copy(), y_train.copy())
X_test_selected_rfe = rfe.transform(X_test_filtered.copy())
selected_features = X_train_selected_rfe.columns.tolist()
print("Features to Remove:")
print(rfe.features_to_remove_)

# Path to save the selected features
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)  # Create directory if it doesn't exist
rfe_selected_features_file = os.path.join(logs_dir, 'rfe_selected_features.txt')

# Save selected feature names to a text file
with open(rfe_selected_features_file, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Selected features saved to {rfe_selected_features_file}")

