from constant import (
    TARGET_MAPPING
)
from utils.utils import match_features
from sklearn.model_selection import train_test_split
import shap
import pandas as pd
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

# Load the saved data cleaner and clean the training and test data

data_cleaner = joblib.load('logs/data_cleaner.pkl')
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

# Define the path to the file containing selected features from a previous steps
file_path = "logs/selected_features/rfe_selected_features.txt"

# Open the file and read the selected feature names
with open(file_path, 'r') as file:
    features_list = [line.strip() for line in file.readlines()]

# Match the selected features with the cleaned training data
original_features = match_features(features_list, X_train_cleaned)

### IMPUTE MISSING VALUES & ENCODE CATEGORICAL VARIABLES

# Load the saved pre processing pipline containing imputer and encoder and apply it to training and test data
preprocessing_pipe = joblib.load('logs/preprocessing_pipe.pkl')
X_train_processed = preprocessing_pipe.fit_transform(X_train_cleaned[original_features])
X_test_processed = preprocessing_pipe.transform(X_test_cleaned[original_features])

### MODEL EXPLAINABILITY USING SHAP

# Load the saved calibrated model
loaded_model = joblib.load('logs/calibrated_trained_model.pkl')

# Extract the base model from CalibratedClassifierCV
base_model = loaded_model.calibrated_classifiers_[0].estimator  # This accesses the original RandomForestClassifier

# Create the SHAP explainer for tree-based models (Random Forest)
explainer = shap.TreeExplainer(base_model)

# Calculate SHAP values for all instances
shap_values = explainer.shap_values(X_test_processed)

# For Class 0
shap.summary_plot(shap_values[:, :, 0], X_test_processed, show=False)
plt.title("Poor Class Summary")
plt.savefig('logs/plots/shap_summary_class_0.png', bbox_inches='tight')
plt.close()

# For Class 1
shap.summary_plot(shap_values[:, :, 1], X_test_processed, show=False)
plt.title("Standard Class Summary")
plt.savefig('logs/plots/shap_summary_class_1.png', bbox_inches='tight')
plt.close()

# For Class 2
shap.summary_plot(shap_values[:, :, 2], X_test_processed, show=False)
plt.title("Good Class Summary")
plt.savefig('logs/plots/shap_summary_class_2.png', bbox_inches='tight')
plt.close()

'''
Features with positive SHAP values positively impact the prediction, while those with negative values have a negative impact.
Each point in the plot represents a real sample of X_test. Then, if a feature has a lot of samples with negative shap values 
and of red colour (high feature value) means that a higher value of that feature tend to negatively affect the output.
'''