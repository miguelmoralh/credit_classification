from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Script with the manually selected features neeeded in the data preprocessing
object_to_float = [
    'Annual_Income', 
    'Changed_Credit_Limit', 
    'Outstanding_Debt', 
    'Amount_invested_monthly', 
    'Monthly_Balance'
]

object_to_int = [
    'Age', 
    'Num_of_Loan', 
    'Num_of_Delayed_Payment'
]

time_to_numeric_years = [
    'Credit_History_Age'
]

months_to_numeric = [
    'Month'
]

multi_label_binarizer_features = [
    'Type_of_Loan'
]

ordinal_variables = [
    'Credit_Mix',
    'Payment_of_Min_Amount'
]

categorical_non_ordinal_variables = [
    'Occupation', 
    'Payment_Behaviour'
]

continuous_features = [
    'Age', 
    'Annual_Income', 
    'Monthly_Inhand_Salary', 
    'Num_Bank_Accounts', 
    'Num_Credit_Card', 
    'Interest_Rate', 
    'Num_of_Loan', 
    'Delay_from_due_date',          
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Credit_History_Age', 
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance'
] 

params_rf_rfe = {    
    "n_estimators": 40,
    "max_depth": 4,
    "criterion": "gini", 
    "random_state": 42
}



