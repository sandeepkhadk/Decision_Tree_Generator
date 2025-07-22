from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import COLUMN_TYPES
import pandas as pd


def detect_column_types(X):
    """Identify numeric and categorical columns"""
    numeric_cols = X.select_dtypes(include=COLUMN_TYPES['numeric']).columns.tolist()
    categorical_cols = X.select_dtypes(include=COLUMN_TYPES['categorical']).columns.tolist()
    return numeric_cols, categorical_cols


def create_preprocessor(numeric_cols, categorical_cols, X):
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')) if X[numeric_cols].isnull().any().any()
        else ('passthrough', 'passthrough')
    ])

    categorical_steps = []
    if categorical_cols:
        if X[categorical_cols].isnull().any().any():
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)))

    categorical_transformer = Pipeline(steps=categorical_steps) if categorical_cols else ('passthrough', 'passthrough')

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])


def get_feature_names(preprocessor):
    """Get feature names after preprocessing"""
    feature_names = []

    # Iterate through the transformers in the ColumnTransformer
    for name, transformer, columns in preprocessor.transformers:
        if transformer == 'passthrough':
            # If the transformer is passthrough, just add the original column names
            feature_names.extend(columns)
        else:
            # If the transformer is a Pipeline or an estimator, we need to extract feature names
            if hasattr(transformer, 'get_feature_names_out'):
                # For transformers that have get_feature_names_out method (like OneHotEncoder)
                transformed_feature_names = transformer.get_feature_names_out(input_features=columns)
                feature_names.extend(transformed_feature_names)
            else:
                # If the transformer does not have this method, we can only add the original column names
                feature_names.extend(columns)

        return feature_names
