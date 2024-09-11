# Script with the manually selected features neeeded in the data preprocessing
OBJECT_TO_FLOAT = [
    'Annual_Income', 
    'Changed_Credit_Limit', 
    'Outstanding_Debt', 
    'Amount_invested_monthly', 
    'Monthly_Balance'
]

OBJECT_TO_INT = [
    'Age', 
    'Num_of_Loan', 
    'Num_of_Delayed_Payment'
]

TIME_TO_NUMERIC_YEARS = [
    'Credit_History_Age'
]

MONTHS_TO_NUMERIC = [
    'Month'
]

MULTI_LABEL_BINARIZER_FEATURES = [
    'Type_of_Loan'
]

ORDINAL_VARIABLES = [
    'Credit_Mix',
    'Payment_of_Min_Amount'
]

CATEGORICAL_NON_ORDINAL_VARIABLES = [
    'Occupation', 
    'Payment_Behaviour'
]

CONTINUOUS_FEATURES = [
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

PARAMS_RF_RFE = {    
    "n_estimators": 40,
    "max_depth": 4,
    "criterion": "gini", 
    "random_state": 42
}



