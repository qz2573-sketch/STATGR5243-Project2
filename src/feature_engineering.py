import pandas as pd
import numpy as np
from typing import List, Union, Optional
from src.utils import logger

def create_arithmetic_feature(df: pd.DataFrame, 
                              col1: str, 
                              col2: str, 
                              operation: str, 
                              new_col_name: str) -> pd.DataFrame:
    """
    Creates a new feature by performing arithmetic operations on two columns.
    """
    df_new = df.copy()
    
    try:
        if operation == 'add':
            df_new[new_col_name] = df_new[col1] + df_new[col2]
        elif operation == 'subtract':
            df_new[new_col_name] = df_new[col1] - df_new[col2]
        elif operation == 'multiply':
            df_new[new_col_name] = df_new[col1] * df_new[col2]
        elif operation == 'divide':
            # Handle division by zero
            df_new[new_col_name] = df_new[col1] / df_new[col2].replace(0, np.nan)
    except Exception as e:
        logger.error(f"Error creating arithmetic feature: {e}")
        
    return df_new

def transform_feature(df: pd.DataFrame, 
                      column: str, 
                      transformation: str) -> pd.DataFrame:
    """
    Applies a transformation to a feature.
    """
    df_new = df.copy()
    
    try:
        if transformation == 'log':
            # Add small constant to avoid log(0)
            df_new[f"{column}_log"] = np.log1p(df_new[column])
        elif transformation == 'square':
            df_new[f"{column}_sq"] = np.square(df_new[column])
        elif transformation == 'sqrt':
            df_new[f"{column}_sqrt"] = np.sqrt(df_new[column])
        elif transformation == 'abs':
            df_new[f"{column}_abs"] = np.abs(df_new[column])
        elif transformation == 'binning':
            # Simple equal-width binning (e.g., 5 bins)
            df_new[f"{column}_binned"] = pd.cut(df_new[column], bins=5)
    except Exception as e:
        logger.error(f"Error transforming feature {column}: {e}")
        
    return df_new

def extract_datetime_features(df: pd.DataFrame, 
                              column: str, 
                              features: List[str]) -> pd.DataFrame:
    """
    Extracts datetime components from a datetime column.
    """
    df_new = df.copy()
    
    try:
        # Ensure it's datetime
        if not pd.api.types.is_datetime64_any_dtype(df_new[column]):
            df_new[column] = pd.to_datetime(df_new[column], errors='coerce')
            
        dt_col = df_new[column].dt
        
        if 'year' in features:
            df_new[f"{column}_year"] = dt_col.year
        if 'month' in features:
            df_new[f"{column}_month"] = dt_col.month
        if 'day' in features:
            df_new[f"{column}_day"] = dt_col.day
        if 'weekday' in features:
            df_new[f"{column}_weekday"] = dt_col.weekday
        if 'quarter' in features:
            df_new[f"{column}_quarter"] = dt_col.quarter
            
    except Exception as e:
        logger.error(f"Error extracting datetime features from {column}: {e}")
        
    return df_new

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drops selected columns.
    """
    return df.drop(columns=columns, errors='ignore')
