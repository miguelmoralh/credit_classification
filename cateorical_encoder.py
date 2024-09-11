from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_variables, categorical_variables):
        self.ordinal_variables = ordinal_variables
        self.categorical_variables = categorical_variables
        self.label_encoders = {}
        self.modes = {}
        self.mappings = {}
        self.mappings['Credit_Mix'] = {
            'Bad': 0,
            'Standard': 1,
            'Good': 2,
            '_': -1
        }
        self.mappings['Payment_of_Min_Amount'] = {
            'Yes': 1,
            'No': 0,
            'NM': -1
        }
        
    def fit(self, X, y=None):
         
        # To identify the categorical, ordinal and special variables not encoded features
        self.original_features = X.select_dtypes(include=['object']).columns
        
        # Update iit lists with available features only
        self.check_init_list(X)

        # Fit LabelEncoder for categorical variables
        for feature in self.categorical_variables:
            if feature in X.columns:
                le = LabelEncoder()
                le.fit(X[feature])
                self.label_encoders[feature] = le
                self.modes[feature] = le.transform(X[feature].mode())[0]

        if 'Credit_Mix' in X.columns:
            self.modes['Credit_Mix'] = self.mappings['Credit_Mix'][X['Credit_Mix'].mode()[0]]
        if 'Payment_of_Min_Amount' in X.columns:
            self.modes['Payment_of_Min_Amount'] = self.mappings['Payment_of_Min_Amount'][X['Payment_of_Min_Amount'].mode()[0]]
        
        return self

    def transform(self, X, y=None):

        # Apply LabelEncoder for categorical variables and handle unseen values
        for feature in self.categorical_variables:
            if feature in X.columns:
                le = self.label_encoders[feature]
                X[f'{feature}_encoded'] = X[feature].apply(lambda x: le.transform([x])[0] if x in le.classes_ else self.modes[feature])

        for feature in self.ordinal_variables:
            if feature in X.columns:
                X[f'{feature}_encoded'] = X[feature].map(self.mappings[feature])  # .map function imputes Nan value if the category is not in the mappings dictionary
                X[f'{feature}_encoded'].fillna(self.modes[feature], inplace=True)  # Impute mode for unseen values
                
        # Finally eliminate the original, not encoded categorical columns (always checking they are in the dataframe)
        X.drop(columns=[col for col in self.original_features if col in X.columns], inplace=True)

        
        return X
    
    def check_init_list(self, X):
        features = (X.columns)
        self.ordinal_variables = [c for c in self.ordinal_variables if c in features]
        self.categorical_variables = [c for c in self.categorical_variables if c in features]