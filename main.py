from constant_variables import (
    object_to_float, object_to_int, time_to_numeric_years, months_to_numeric,
    multi_label_binarizer_features, ordinal_variables, 
    categorical_non_ordinal_variables, continuous_features, params_rf_rfe
)
from utils.data_cleanning import DataCleanning
from utils.imputer import ImputeMissing
from utils.cateorical_encoder import CategoricalEncoder
from utils.dependence_feature_selection import FeatureSelection
from utils.rfe_multivariant_feature_selection import CustomRFECV
from hyperparams_optimization import objective, get_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
import functools

# Load dataframe
df = pd.read_csv('data/train.csv')

# Separate the target from the dataframe
x = df.drop(columns=['Credit_Score'])
y = df['Credit_Score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

### DATASET CLEANNING
data_cleaner = DataCleanning(
    object_to_float=object_to_float,                      
    object_to_int=object_to_int,
    time_to_numeric_years=time_to_numeric_years, 
    months_to_numeric=months_to_numeric,
    multi_label_binarizer_features=multi_label_binarizer_features,
    nan_threshold = 0.9,
    ids_threshold = 0.12, 
    unique_threshold = 1
)
X_train_cleaned = data_cleaner.fit_transform(X_train.copy())
X_test_cleaned = data_cleaner.transform(X_test.copy())

### IMPUTE MISSING VALUES
imputer = ImputeMissing()  
X_train_imputed = imputer.fit_transform(X_train_cleaned)
X_test_imputed = imputer.transform(X_test_cleaned)

### ENCODE CATEGORICAL VARIABLES
encoder = CategoricalEncoder(
    ordinal_variables=ordinal_variables, 
    categorical_variables=categorical_non_ordinal_variables
)
X_train_encoded = encoder.fit_transform(X_train_imputed)
X_test_encoded = encoder.transform(X_test_imputed)


