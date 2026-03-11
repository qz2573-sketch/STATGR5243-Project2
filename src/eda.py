import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Union
from src.utils import logger

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns summary statistics for numeric columns.
    """
    return df.describe().T.reset_index().rename(columns={'index': 'variable'})

def get_categorical_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Returns frequency summary for a categorical column.
    """
    return df[column].value_counts().reset_index().rename(columns={'index': column, column: 'count'})

def plot_histogram(df: pd.DataFrame, x: str, bins: int = 30, color: Optional[str] = None):
    """
    Creates a histogram.
    """
    try:
        fig = px.histogram(df, x=x, nbins=bins, color=color, title=f"Histogram of {x}")
        return fig
    except Exception as e:
        logger.error(f"Error creating histogram: {e}")
        return None

def plot_box(df: pd.DataFrame, y: str, x: Optional[str] = None, color: Optional[str] = None):
    """
    Creates a box plot.
    """
    try:
        fig = px.box(df, y=y, x=x, color=color, title=f"Box Plot of {y}")
        return fig
    except Exception as e:
        logger.error(f"Error creating box plot: {e}")
        return None

def plot_bar(df: pd.DataFrame, x: str, y: Optional[str] = None, color: Optional[str] = None, aggregation: str = 'count'):
    """
    Creates a bar chart.
    """
    try:
        if y is None:
            # Count plot
            counts = df[x].value_counts().reset_index()
            counts.columns = [x, 'count']
            fig = px.bar(counts, x=x, y='count', color=color, title=f"Count of {x}")
        else:
            # Aggregated bar chart
            if aggregation == 'mean':
                df_agg = df.groupby(x)[y].mean().reset_index()
                title = f"Mean of {y} by {x}"
            elif aggregation == 'sum':
                df_agg = df.groupby(x)[y].sum().reset_index()
                title = f"Sum of {y} by {x}"
            else:
                df_agg = df # Raw values if applicable
                title = f"{y} by {x}"
                
            fig = px.bar(df_agg, x=x, y=y, color=color, title=title)
        return fig
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        return None

def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None):
    """
    Creates a scatter plot.
    """
    try:
        fig = px.scatter(df, x=x, y=y, color=color, size=size, title=f"Scatter Plot: {y} vs {x}")
        return fig
    except Exception as e:
        logger.error(f"Error creating scatter plot: {e}")
        return None

def plot_heatmap(df: pd.DataFrame, columns: List[str]):
    """
    Creates a correlation heatmap.
    """
    try:
        corr = df[columns].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        return fig
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return None
