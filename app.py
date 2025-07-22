import streamlit as st
import pandas as pd
from preprocessing import detect_column_types, create_preprocessor, get_feature_names
from visualization import plot_pairplot, plot_correlation_heatmap
from model import determine_problem_type, create_model, evaluate_model
from config import PLOT_CONFIG
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.pipeline import Pipeline
import numpy as np

# Main application layout
def main():
    st.set_page_config(layout="wide", page_title="Smart Decision Tree Builder")
    st.title("🌳 Smart Decision Tree Builder with Feature Selector")

    # File upload and basic setup
    uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx"])
    if not uploaded_file:
        return st.info("👋 Please upload a dataset to get started")

    try:
        # Load data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Select target variable
        target_col = st.selectbox("🎯 Select target variable", options=df.columns)
        y = df[target_col]

        # Data preprocessing
        X = df.drop(columns=[target_col])
        numeric_cols, categorical_cols = detect_column_types(X)
        preprocessor = create_preprocessor(numeric_cols, categorical_cols, X)

        # Problem type determination
        problem_type = determine_problem_type(y)
        st.info(f"🔮 Detected problem type: {problem_type.capitalize()}")

        # Data exploration section
        st.subheader("🔍 Data Exploration")
        with st.expander("Data Preview"):
            st.dataframe(df.head())

            tab1, tab2, tab3 = st.tabs(["Statistics", "Missing Values", "Visualizations"])

            with tab1:
                st.write(df.describe(include='all'))

            with tab2:
                st.write(df.isnull().sum())

            with tab3:
                viz_type = st.radio("Visualization Type:", ["Pair Plot", "Correlation", "Target Relationship"])

                if viz_type == "Pair Plot":
                    sample_size = st.slider("Sample size", 50, len(df), PLOT_CONFIG['pairplot_sample_size'])
                    cols_to_plot = st.multiselect("Select columns", numeric_cols + categorical_cols, default=numeric_cols[:3] if numeric_cols else [])
                    if cols_to_plot:
                        fig = plot_pairplot(df, cols_to_plot, target_col, sample_size)
                        st.pyplot(fig)

                elif viz_type == "Correlation" and numeric_cols:
                    fig = plot_correlation_heatmap(df, numeric_cols + [target_col])
                    st.pyplot(fig)

                elif viz_type == "Target Relationship":
                    col = st.selectbox("Select feature", numeric_cols + categorical_cols)
                    if col:
                        st.scatter_chart(df[[col, target_col]])

        # Feature selection
        st.subheader("🧐 Feature Selection")
        cols_to_include = st.multiselect("Select features to include:", options=numeric_cols + categorical_cols, default=numeric_cols + categorical_cols)

        if not cols_to_include:
            st.warning("Please select at least one feature")
            st.stop()

        # Update features
        X = X[cols_to_include]
        numeric_cols = [col for col in numeric_cols if col in cols_to_include]
        categorical_cols = [col for col in categorical_cols if col in cols_to_include]

        # Update the preprocessor after feature selection
        preprocessor = create_preprocessor(numeric_cols, categorical_cols, X)

        # Parameters for model creation
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
        max_depth = st.slider("Max Depth", 1, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Create the model
        model_params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': 42
        }

        # Create and fit pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', create_model(problem_type, model_params))
        ])
        pipeline.fit(X_train, y_train)

        # Get feature names after preprocessing
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = get_feature_names(preprocessor)  # Fallback if not using ColumnTransformer

        # Model evaluation
        st.subheader("📊 Model Performance")
        train_metrics = evaluate_model(pipeline, X_train, y_train, problem_type)
        test_metrics = evaluate_model(pipeline, X_test, y_test, problem_type)

        # Visualization
        st.subheader("🌲 Decision Tree Visualization")
        model = pipeline.named_steps['model']  # Extract the model from the pipeline
        if problem_type == "classification":
            class_names = [str(cls) for cls in y.unique()]
        else:
            class_names = None

        dot_data = export_graphviz(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True
        )
        st.graphviz_chart(dot_data)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
