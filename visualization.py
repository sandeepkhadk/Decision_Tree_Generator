import matplotlib.pyplot as plt
import seaborn as sns
from config import PLOT_CONFIG

def plot_pairplot(df, cols, target_col, sample_size):
    """Generate pairplot for selected columns"""
    fig = sns.pairplot(df.sample(sample_size).dropna(),
                      vars=cols, hue=target_col)
    return fig.fig

def plot_correlation_heatmap(df, numeric_cols):
    """Generate correlation heatmap"""
    plt.figure(figsize=PLOT_CONFIG['heatmap_figsize'])
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    return plt

def plot_feature_importance(importance_df):
    """Plot feature importance"""
    plt.figure(figsize=PLOT_CONFIG['feature_importance_figsize'])
    sns.barplot(x='Importance', y='Feature',
               data=importance_df.head(20))
    plt.title('Top 20 Features by Mutual Information')
    return plt
