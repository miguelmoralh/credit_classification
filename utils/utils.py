def match_features(features_list, df):
    """
    Return a list of the common features between a dataframe and a list of features,
    accounting for potential encoding differences in column names.

    This function compares the feature names in the input list (`features_list`) with the
    column names of the dataframe (`df`). If a feature from the list is present in the
    dataframe, it is added to the `matched_features` list. Additionally, if a feature in the
    list has an '_encoded' suffix but its base name (without the suffix) exists in the dataframe,
    the base name is considered a match and added to the `matched_features`.

    Args:
        features_list (list): A list of feature names (strings) to check for matches.
        df (pd.DataFrame): The dataframe containing columns to match the features against.

    Returns:
        list: A list of matched feature names from the dataframe.
    """
    matched_features = []
    for feature in features_list:
        if feature in df.columns:
            matched_features.append(feature)
        elif feature.endswith('_encoded'):
            base_feature = feature[:-8]  # Remove '_encoded' suffix
            if base_feature in df.columns:
                matched_features.append(base_feature)
    return matched_features





