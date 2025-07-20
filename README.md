#  🌳 Decision Tree Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

An interactive web application for building, visualizing, and analyzing decision trees from uploaded datasets.


## Features

-  📊 Upload CSV/Excel datasets
-  🔍 Interactive data exploration
-  🌳 Customizable decision tree parameters
-  📈 Feature importance visualization
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

### Running the Application
   ***Start the Streamlit app:***
   
        streamlit run main.py
### Project Structure
      decision_tree_app/
      ├── main.py                 # Main application entry point
      ├── preprocessing.py        # Data preprocessing functions
      ├── visualization.py        # Visualization utilities
      ├── model.py                # Model building and evaluation
      ├── config.py               # Configuration constants

### Usage Guide
   1. Upload your dataset (CSV or Excel format)
   2. Select the target variable you want to predict
   3. Explore your data using interactive visualizations
   4. Select features to include in the model
   5. Adjust tree parameters in the sidebar
   6. View model performance metrics
   7. Visualize the decision tree and feature importance
   8. Export results as needed



 
