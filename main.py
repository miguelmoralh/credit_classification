
from constant import (
    OBJECT_TO_FLOAT, OBJECT_TO_INT, TIME_TO_NUMERIC_YEARS, MONTHS_TO_NUMERIC,
    MULTI_LABEL_BINARIZER_FEATURES, ORDINAL_VARIABLES, 
    CATEGORICAL_NON_ORDINAL_VARIABLES, TARGET_MAPPING
)
from utils.utils import match_features
from utils.utils_hyp_opt import find_best_trial, get_best_model, optimize_calibration_multiclass
from data_cleanning import DataCleanning
from imputer import ImputeMissing
from cateorical_encoder import CategoricalEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

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

# Save the data cleaner instance as a pickle file
logs_dir = "logs"
data_cleaner_filename = os.path.join(logs_dir, 'data_cleaner.pkl')
joblib.dump(data_cleaner, data_cleaner_filename)

# Define the path to the file containing selected features from a previous steps
file_path = "logs/selected_features/rfe_selected_features.txt"

# Open the file and read the selected feature names
with open(file_path, 'r') as file:
    features_list = [line.strip() for line in file.readlines()]

# Match the selected features with the cleaned training data
original_features = match_features(features_list, X_train_cleaned)

### IMPUTE MISSING VALUES & ENCODE CATEGORICAL VARIABLES  

imputer = ImputeMissing()
encoder = CategoricalEncoder(
    ordinal_variables=ORDINAL_VARIABLES, 
    categorical_variables=CATEGORICAL_NON_ORDINAL_VARIABLES
)

# Create a pipeline to apply the imputer and encoder sequentially
preprocessing_pipe = Pipeline(
    [
        ('imputer', imputer), 
        ('cat_encoder', encoder)
    ]
)

# Preprocess the training and test data
X_train_processed = preprocessing_pipe.fit_transform(X_train_cleaned[original_features])
X_test_processed = preprocessing_pipe.transform(X_test_cleaned[original_features])

# Save the preprocessing pipeline as a pickle file
preprocessing_pipe_filename = os.path.join(logs_dir, 'preprocessing_pipe.pkl')
joblib.dump(preprocessing_pipe, preprocessing_pipe_filename)

### TRAIN THE MODEL

# Find the optimal (BEST) trial
logs_dir_trials = 'logs/optuna_trials' 
best_trial_info = find_best_trial(logs_dir_trials, overfitting_threshold=0.02)

if best_trial_info:
    print(f"Best model: {best_trial_info['model']}")
    print(f"Trial number: {best_trial_info['trial_number']}")
    print(f"ROC AUC: {best_trial_info['roc_auc']}")
    print(f"Overfitting: {best_trial_info['overfitting']}")
    print(f"Hyperparameters: {best_trial_info['params']}")
else:
    print("No trail has been found below the overfitting thershold.")
    
# Get the model and its hyperparameters
model = get_best_model(best_trial_info)

# Train the model
model.fit(X_train_processed, y_train)

### PROBABILITY CALIBRATION

# Optimize the calibration method for the trained model using training data
best_method, brier_score = optimize_calibration_multiclass(
    model, X_train_processed, y_train
)
print(f"The best calibration method is: {best_method} with Brier Score: {brier_score}")

# Calibrate the model 
calibrated_model = CalibratedClassifierCV(model, method=best_method, cv='prefit') # Prefit to indicate that the estimator is already fitted (all data used for callibration)
calibrated_model.fit(X_train_processed, y_train)

# Save the calibrated model to a pickle file
model_filename = os.path.join(logs_dir, 'calibrated_trained_model.pkl')
joblib.dump(calibrated_model, model_filename)

### VISUALIZATION OF CALIBRATED CURVE

# Predict calibrated probabilities
y_prob_train_calibrated = calibrated_model.predict_proba(X_train_processed)
y_prob_test_calibrated = calibrated_model.predict_proba(X_test_processed)

# Generate calibration curve
prob_true_calibrated, prob_pred_calibrated = calibration_curve((y_test == 0).astype(int), y_prob_test_calibrated[:, 0], n_bins=10)

# path to save the plot
output_dir = 'logs/plots'

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









