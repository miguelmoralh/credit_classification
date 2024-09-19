from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, CONTINUOUS_FEATURES, PARAMS_RF_RFE
)
from utils.utils import match_features, optimize_calibration_multiclass
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np



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

### RESULTS AND PREDICTIONS

# Predict calibrated probabilities
y_prob_train = loaded_model.predict_proba(X_train_encoded)
y_prob_test = loaded_model.predict_proba(X_test_encoded)

# Evaluate the model's performance using ROC AUC with ovo (one-vs-one)
roc_auc_train = roc_auc_score(y_train, y_prob_train, multi_class='ovo')
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovo')

print(f"Train ROC AUC Score: {roc_auc_train}")
print(f"Test ROC AUC Score: {roc_auc_test}")

# Predict the correct label
y_pred_train = loaded_model.predict(X_train_encoded)
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