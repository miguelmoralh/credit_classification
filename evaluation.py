from constant import (
  TARGET_MAPPING
)
from utils.utils import match_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

### RESULTS AND PREDICTIONS

# Load the saved calibrated model
loaded_model = joblib.load('logs/calibrated_trained_model.pkl')

# Predict calibrated probabilities
y_prob_train = loaded_model.predict_proba(X_train_processed)
y_prob_test = loaded_model.predict_proba(X_test_processed)

# Evaluate the model's performance using ROC AUC with ovo (one-vs-one)
roc_auc_train = roc_auc_score(y_train, y_prob_train, multi_class='ovo')
roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class='ovo')

print(f"Train ROC AUC Score: {roc_auc_train}")
print(f"Test ROC AUC Score: {roc_auc_test}")

# Predict the correct label
y_pred_train = loaded_model.predict(X_train_processed)
y_test_pred = loaded_model.predict(X_test_processed)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Convert to percentages
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Define the labels
labels = ['Poor', 'Standard', 'Good']

# Create the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(conf_matrix_percent, cmap='Blues', 
                 xticklabels=labels, yticklabels=labels, vmin=0, vmax=100,
                 annot=False)  # Turn off Seaborn's automatic annotations

# Manually add text annotations to each cell
for i in range(conf_matrix_percent.shape[0]):
    for j in range(conf_matrix_percent.shape[1]):
        ax.text(j+0.5, i+0.5, f'{conf_matrix_percent[i, j]:.1f}',
                ha="center", va="center", color="black")

# Customize the plot
plt.title('Confusion Matrix (Percentages)')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Adjust layout to prevent cutoff
plt.tight_layout()

# Save and plot
plt.savefig('logs/plots/confusion_matrix_percent.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and print classification report
print(classification_report(y_test, y_test_pred, target_names=['Poor', 'Standard', 'Good']))

