from constant_variables import (
    object_to_float, object_to_int, time_to_numeric_years, months_to_numeric,
    multi_label_binarizer_features, ordinal_variables, 
    categorical_non_ordinal_variables, params_rf_rfe
)
from utils.data_cleanning import DataCleanning
from utils.imputer import ImputeMissing
from utils.cateorical_encoder import CategoricalEncoder
from utils.rfe_multivariant_feature_selection import CustomRFECV
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

# Initialize imputer and encoder
imputer = ImputeMissing()  
encoder = CategoricalEncoder(
    ordinal_variables=ordinal_variables, 
    categorical_variables=categorical_non_ordinal_variables
)

### MULTIVARIANT FEATURE SELECTION : RECURSIVE FEATURE ELIMINATION (RFE)

# Use Random Forest as model 
rf_clf = RandomForestClassifier(**params_rf_rfe)  # ** to initialize model from dict 
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

X_train_selected_rfe = rfe.fit_transform(X_train_cleaned.copy(), y_train.copy())
X_test_selected_rfe = rfe.transform(X_test_cleaned.copy())
print("Features to Remove:")
print(rfe.features_to_remove_)