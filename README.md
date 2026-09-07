# ML Trees Generator

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An interactive Streamlit app for uploading a dataset, profiling the data, training a tree-based model, visualizing the result, and exporting downloadable reports.

Live demo: https://mltreestudio.streamlit.app/

## Features

- CSV and Excel upload with validation and friendly error handling
- Automatic dataset summary with rows, columns, missing values, duplicates, and preview
- Target selection and feature selection with automatic numeric and categorical detection
- Automatic preprocessing for missing values and categorical features before training
- Downloadable dataset profile JSON, trained model bundle, and PDF result report
- Responsive layout tuned for smaller screens
- Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, XGBoost, LightGBM, and CatBoost configuration
- Classification and regression metrics
- Tree visualization with optional depth limiting and image download when supported
- Feature importance visualization for tree-based models
- Dynamic prediction form based on the selected features

## Data Cleaning and Preprocessing

The app handles some common messy-data cases automatically during training:

- Missing numeric values are imputed with the median
- Missing categorical values are imputed with the most frequent value
- Categorical features are one-hot encoded for model training
- Unsupported column types are ignored with a warning

It does not perform full data cleaning such as fixing inconsistent labels, parsing malformed dates, or removing duplicate rows automatically, so very messy datasets may still need manual cleanup first.

## Project Structure

```text
Decision_Tree_Generator/
├── app.py
├── config.py
├── data_utils.py
├── model.py
├── model_utils.py
├── preprocessing.py
├── report_utils.py
├── tree_utils.py
├── ui_utils.py
├── visualization.py
├── requirements.txt
├── .devcontainer/
│   └── devcontainer.json
└── .streamlit/
   └── config.toml
```

## Architecture Summary

- `app.py` is the Streamlit entry point and controls the dashboard flow.
- `ui_utils.py` handles page rendering, session state, and interactive controls.
- `data_utils.py` handles upload validation, dataset loading, and summary generation.
- `preprocessing.py` detects column types and builds the preprocessing pipeline.
- `model_utils.py` creates models and computes evaluation metrics.
- `report_utils.py` builds downloadable reports and reusable result visualizations.
- `tree_utils.py` renders trained tree-based models when visualization is supported.
- `config.py` stores environment-driven defaults and UI settings.

## Requirements

- Python 3.11 or newer
- pip package manager

## Installation

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it on Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Local Development

Run the application with:

```bash
streamlit run app.py
```

Optional environment variables:

- `APP_NAME`
- `APP_TAGLINE`
- `MAX_UPLOAD_MB`
- `DEFAULT_TEST_SIZE`
- `DEFAULT_RANDOM_STATE`
- `DEFAULT_MAX_DEPTH`
- `DEFAULT_MIN_SAMPLES_SPLIT`
- `DEFAULT_MIN_SAMPLES_LEAF`

## Usage

1. Upload a CSV or Excel dataset.
2. Review the dataset profile in the explorer section.
3. Select the target column and feature columns.
4. Configure the tree-based model parameters.
5. Train the model.
6. Review the metrics and visualization.
7. Use the prediction form to generate outputs for new records.

## Model Details

The app trains a selected tree-based model, including `DecisionTreeClassifier` / `DecisionTreeRegressor`, `RandomForestClassifier` / `RandomForestRegressor`, `ExtraTreesClassifier` / `ExtraTreesRegressor`, `GradientBoostingClassifier` / `GradientBoostingRegressor`, `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`, `AdaBoostClassifier` / `AdaBoostRegressor`, `XGBClassifier` / `XGBRegressor`, `LGBMClassifier` / `LGBMRegressor`, and `CatBoostClassifier` / `CatBoostRegressor` depending on the chosen model and target column. Numeric features are imputed with the median, categorical features are imputed with the most frequent value and one-hot encoded, and the train/test split is reproducible through a configurable random state.

## Evaluation Metrics

Classification:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

Regression:

- MAE
- MSE
- RMSE
- R²

## Deployment Instructions

Recommended platform: Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Point the app to `app.py`.
4. Ensure `requirements.txt` is present at the repository root.
5. Deploy the app.

For a local smoke test before deployment:

```bash
streamlit run app.py
```





