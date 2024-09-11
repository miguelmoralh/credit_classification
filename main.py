
from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, CONTINUOUS_FEATURES, PARAMS_RF_RFE
)
from utils.utils import match_features, find_best_trial, get_best_model
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

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

    
'''
For each model I have saved all the optuna hyperparameters optimization results in json files inside logs folder.
Remember that I have been optimizing trying to get the maximum ROC AUC score and the minimum overfitting score. 
Now it is time to select which model and with which hyperparameters will be the one we will train and evaluate. 
To make the choice I have decided to a threshold for the overfitting score of 0.02. So, I will look to all the trials
of all models that have an overfitting below 0.02 and I will choose the trial that has the highest ROC AUC score. 
 
'''

# Use the function 'find_best_trial' to find the OPTIMAL (BEST) TRIAL
logs_dir = 'logs' 
best_trial_info = find_best_trial(logs_dir)

if best_trial_info:
    print(f"Best model: {best_trial_info['model']}")
    print(f"Trial number: {best_trial_info['trial_number']}")
    print(f"ROC AUC: {best_trial_info['roc_auc']}")
    print(f"Overfitting: {best_trial_info['overfitting']}")
    print(f"Hyperparameters: {best_trial_info['params']}")
else:
    print("No trail has been found below the overfitting thershold.")
    
### TRAIN THE MODEL

# Get the model and its hyperparams from the 'best_trial_info'
model = get_best_model(best_trial_info)

# Train the model on the full training set
model.fit(X_train_encoded, y_train)

### SAVE THE MODEL TO A PICKLE FILE IN THE 'logs' FOLDER

os.makedirs(logs_dir, exist_ok=True)
model_filename = os.path.join(logs_dir, 'trained_model.pkl')
joblib.dump(model, model_filename)

### PREDICTIONS AND RESULTS

#Load the saved model (to ensure it's properly saved)
loaded_model = joblib.load('logs/trained_model.pkl')

# Predict probabilities for each label
y_pred_train_proba = loaded_model.predict_proba(X_train_encoded)
y_test_pred_proba = loaded_model.predict_proba(X_test_encoded)

# Evaluate the model's performance using ROC AUC with ovo (one-vs-one)
roc_auc_train = roc_auc_score(y_train, y_pred_train_proba, multi_class='ovo')
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovo')

print(f"Train ROC AUC Score: {roc_auc_train}")
print(f"Test ROC AUC Score: {roc_auc_test}")

# Generate class predictions from the model
y_train_pred = loaded_model.predict(X_train_encoded)
y_test_pred = loaded_model.predict(X_test_encoded)

# Create the confusion matrix for the test set
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Define the labels
labels = ['Poor', 'Standard', 'Good']

# Create the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

# Customize the plot
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Ensure the 'logs' folder exists
os.makedirs('logs', exist_ok=True)

# Save and plot
plt.savefig('logs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print classification report
print(classification_report(y_test, y_test_pred, target_names=['Poor', 'Standard', 'Good']))




