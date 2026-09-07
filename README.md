#  🌳 ML Trees Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mltreestudio.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

An interactive dashboard-style web application for building, configuring, and visualizing tree-based machine learning models — decision trees, random forests, extra trees, gradient boosting, hist gradient boosting, AdaBoost, XGBoost, LightGBM, and CatBoost — from uploaded datasets.

Live demo: https://mltreestudio.streamlit.app/


## Features

-  🏠 Dashboard home page with quick actions and model cards
-  🧭 Sidebar navigation across all supported models and the dataset page
-  📊 Upload CSV/Excel datasets with a drag-and-drop interface
-  🔍 Interactive data exploration
-  🌲 Multiple tree-based model support, each with its own configuration and description
-  📈 Feature importance visualization
-  🌳 Interactive tree viewer with zoom, pan, fit-to-view, and fullscreen controls
-  🎯 Automatic problem type detection (classification/regression)
-  💾 Export decision trees and feature importance

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sandeepkhadk/Decision_Tree_Generator.git
   cd Decision_Tree_Generator
   
2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**
    ```bash
   pip install -r requirements.txt

# ML Trees Generator

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An interactive Streamlit application for uploading a dataset, profiling the data, training a tree-based model, visualizing the tree when supported, evaluating performance, and generating predictions in the browser.

## Features

- CSV and Excel upload with validation and friendly error handling
- Automatic dataset summary with rows, columns, missing values, duplicates, and preview
- Target selection and feature selection with automatic numeric/categorical detection
- Automatic preprocessing for missing values and categorical features before model training
- Reproducible preprocessing with imputation and one-hot encoding
- Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, and AdaBoost configuration
- Classification and regression metrics
- Tree visualization with optional depth limiting and image download when supported
- Feature importance visualization for tree-based models
- Dynamic prediction form based on the selected features

## Screenshots

Add screenshots of the dashboard here after deployment.

## Technology Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL
- XGBoost
- LightGBM
- CatBoost

## Architecture Summary

- `app.py` is the Streamlit entry point and controls the full dashboard flow.
- `data_utils.py` handles upload validation, dataset loading, and summary generation.
- `preprocessing.py` detects column types and builds the preprocessing pipeline.
- `model_utils.py` creates the model and computes metrics.
- `tree_utils.py` renders the trained tree-based model when visualization is supported.
- `config.py` stores environment-driven defaults and UI settings.

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
pip install -r requirements.txt
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

## Data Cleaning and Preprocessing

The app handles some common messy-data cases automatically during training:

- Missing numeric values are imputed with the median.
- Missing categorical values are imputed with the most frequent value.
- Categorical features are one-hot encoded for model training.
- Unsupported column types are ignored with a warning.

It does not perform full data cleaning such as fixing inconsistent labels, parsing malformed dates, or removing duplicate rows automatically, so very messy datasets may still need manual cleanup first.

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

## Project Structure

```text
Decision_Tree_Generator/
├── app.py
├── config.py
├── data_utils.py
├── model_utils.py
├── preprocessing.py
├── tree_utils.py
├── visualization.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

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

## Future Improvements

- Add exportable model artifacts
- Add per-session model persistence
- Add schema validation and dataset profiling reports





