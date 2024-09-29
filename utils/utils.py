def match_features(features_list, df):
    """
    Return a list of the common features between a dataframe and a list of columns taking 
    the differences in the names as ones appear encoded and other not.

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





