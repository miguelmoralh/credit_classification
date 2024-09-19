from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, 
)
from utils.utils import match_features, optimize_calibration_multiclass
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
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

### PROBABILITY CALIBRATION

#Load the saved model (to ensure it's properly saved)
loaded_model = joblib.load('logs/trained_model.pkl')

# Call the function to get the best calibration method
best_method, brier_score = optimize_calibration_multiclass(
    loaded_model, X_train_encoded, y_train
)

# Print out the best method and its Brier score
print(f"The best calibration method is: {best_method} with Brier Score: {brier_score}")

# Calibrate the model 
calibrated_model = CalibratedClassifierCV(loaded_model, method=best_method, cv='prefit') # Prefit to indicate that the estimator is already fitted (all data used for cllibration)
calibrated_model.fit(X_train_encoded, y_train)

# save the calibrated model to a pickle file
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
model_filename = os.path.join(logs_dir, 'calibrated_trained_model.pkl')
joblib.dump(calibrated_model, model_filename)

### VISUALIZATION OF CALIBRATED CURVE

# Predict calibrated probabilities
y_prob_train_calibrated = calibrated_model.predict_proba(X_train_encoded)
y_prob_test_calibrated = calibrated_model.predict_proba(X_test_encoded)

# Generate calibration curve
prob_true_calibrated, prob_pred_calibrated = calibration_curve((y_test == 0).astype(int), y_prob_test_calibrated[:, 0], n_bins=10)

# Visualize calibration curve of best model

# Make sure 'logs' folder exists
output_dir = 'logs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Complete path where it is going to be saved
output_file = os.path.join(output_dir, 'calibration_curve.png')

# Visualize calibration curve of best mode
plt.figure(figsize=(10, 6))
plt.plot(prob_pred_calibrated, prob_true_calibrated, label='{} (Brier={:.4f})'.format(best_method, brier_score))
plt.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves for Random Forest')
plt.legend(loc='best')
plt.savefig(output_file) # Save plot
plt.show()



