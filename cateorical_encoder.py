from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom transformer for encoding categorical and ordinal variables using LabelEncoder 
    for categorical variables and custom mappings for ordinal variables.
    
    This class also handles unseen values during transformation by replacing them with 
    the mode (for categorical variables) or predefined mappings (for ordinal variables).
    
    Inherits from sklearn's BaseEstimator and TransformerMixin for compatibility 
    with scikit-learn's pipeline.
    """
    def __init__(self, ordinal_variables, categorical_variables):
        """
        Initialize the CategoricalEncoder with the given categorical and ordinal variables.
        
        Args:
            ordinal_variables (list): List of ordinal variables to encode.
            categorical_variables (list): List of categorical variables to encode using LabelEncoder.
        """
        self.ordinal_variables = ordinal_variables
        self.categorical_variables = categorical_variables
        self.label_encoders = {} # Dictionary to store fitted LabelEncoders for categorical variables
        self.modes = {} # Dictionary to store mode values for handling unseen categorical data
        self.mappings = {} # Dictionary to store mappings for ordinal variables
        
        # Custom mappings for ordinal variables 'Credit_Mix' and 'Payment_of_Min_Amount'
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
        """
        Fit the transformer by creating LabelEncoders for categorical variables and
        storing mode values to handle unseen categories during transformation.
        
        Args:
            X (pd.DataFrame): Input dataset to fit the transformer on.
            y (pd.Series, optional): Target variable, not used here.
        
        Returns:
            self: Fitted transformer with initialized encoders and mode values.
        """
        # Identify the original categorical features in the dataset
        self.original_features = X.select_dtypes(include=['object']).columns
        
       # Update the lists of ordinal and categorical variables to include only the features present in the dataset.
        self.check_init_list(X)

        # Fit LabelEncoder for each categorical variable
        for feature in self.categorical_variables:
            if feature in X.columns:
                le = LabelEncoder() # Initialize LabelEncoder
                le.fit(X[feature])
                self.label_encoders[feature] = le # Store fitted LabelEncoder
                self.modes[feature] = le.transform(X[feature].mode())[0] # Store the mode value for handling unseen categories

         # Handle modes for specific ordinal variables
        if 'Credit_Mix' in X.columns:
            self.modes['Credit_Mix'] = self.mappings['Credit_Mix'][X['Credit_Mix'].mode()[0]]
        if 'Payment_of_Min_Amount' in X.columns:
            self.modes['Payment_of_Min_Amount'] = self.mappings['Payment_of_Min_Amount'][X['Payment_of_Min_Amount'].mode()[0]]
        
        return self

    def transform(self, X, y=None):
        """
        Transform the dataset by encoding categorical and ordinal variables.
        
        Args:
            X (pd.DataFrame): Input dataset to transform.
            y (pd.Series, optional): Target variable, not used here.
        
        Returns:
            pd.DataFrame: Transformed dataset with encoded variables and original categorical variables dropped.
        """

        # Apply LabelEncoder for categorical variables and handle unseen values using precomputed mode
        for feature in self.categorical_variables:
            if feature in X.columns:
                le = self.label_encoders[feature]
                X[f'{feature}_encoded'] = X[feature].apply(lambda x: le.transform([x])[0] if x in le.classes_ else self.modes[feature])

        # Apply custom mappings for ordinal variables and fill unseen/missing values using precomputed mode
        for feature in self.ordinal_variables:
            if feature in X.columns:
                X[f'{feature}_encoded'] = X[feature].map(self.mappings[feature])  # imputes Nan value if the category isn't in the mappings dictionary
                X[f'{feature}_encoded'].fillna(self.modes[feature], inplace=True)
                
        # Drop the original categorical columns after encoding
        X.drop(columns=[col for col in self.original_features if col in X.columns], inplace=True)

        return X
    
    def check_init_list(self, X):
        """
        Function to ensure that only variables present in the dataset are considered
        for encoding. Updates the lists of ordinal and categorical variables accordingly.
        
        Args:
            X (pd.DataFrame): Input dataset.
        """
        features = (X.columns)
        self.ordinal_variables = [c for c in self.ordinal_variables if c in features]
        self.categorical_variables = [c for c in self.categorical_variables if c in features]