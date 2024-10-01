# Credit Classification Project
This project tackles a multi-class credit classification task using historical loan data. The pipeline includes data cleaning, imputation of missing values, encoding categorical variables, feature selection (using Bivariate Dependence Feature Selection using Normalized Mutual Info and Recursive Feature Elimination), hyperparameter optimization with Optuna, model training, and probability calibration.

The model is evaluated using confusion matrix and explained using SHAP for feature importance insights. The pipeline is adaptable to other datasets with minor modifications.

To train the model, simply run main.py after configuring your dataset. All feature selection, model optimization, and calibration decisions are based on the training set, ensuring no data leakage.

## Scripts:
- data_cleaning.py: Class that cleans the dataset (type conversion, feature removal).
- imputer.py: Class to handle missing values (median for numeric, "Missing" for categorical).
- categorical_encoder.py: Class to encode categorical features (manual mapping for ordinal, LabelEncoder for non ordinal).
- 01_main_dependence_fs.py: Bivariate feature selection using Normalized Mutual Information.
- 02_main_rfe_fs.py: Custom Recursive Feature Elimination with cross-validation.
- 03_main_hyp_opt.py: Hyperparameter and model optimization using Optuna.
- main.py: Trains and calibrates the model.
- evaluation.py: Evaluates the trained model (ROC AUC, confusion matrix).
- model_explainability.py: Explains model predictions using SHAP values.
