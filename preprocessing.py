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
        ('imputer', SimpleImputer(strategy='median'))
    ]) if numeric_cols else 'passthrough'

    categorical_transformer = 'passthrough'
    if categorical_cols:
        try:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', encoder)
        ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])


def get_feature_names(preprocessor):
    """Get feature names after preprocessing"""
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    feature_names = []

    for name, transformer, columns in preprocessor.transformers:
        if transformer == 'passthrough':
            feature_names.extend(columns)
        else:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(input_features=columns))
            elif hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                onehot = transformer.named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    feature_names.extend(onehot.get_feature_names_out(input_features=columns))
                else:
                    feature_names.extend(columns)
            else:
                feature_names.extend(columns)

    return feature_names
