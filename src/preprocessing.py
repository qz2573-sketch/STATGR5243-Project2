import pandas as pd
import numpy as np
from typing import List, Union, Optional
from src.utils import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def handle_missing_values(df: pd.DataFrame, 
                          method: str = 'drop_rows', 
                          columns: Optional[List[str]] = None,
                          fill_value: Optional[Union[str, float]] = None) -> pd.DataFrame:
    """
    Handles missing values in the dataframe.
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns
        
    if method == 'drop_rows':
        df_clean = df_clean.dropna(subset=columns)
    elif method == 'drop_cols':
        # Drop columns if they have any missing values (or maybe a threshold, but simplistic for now)
        cols_to_drop = [c for c in columns if df_clean[c].isnull().any()]
        df_clean = df_clean.drop(columns=cols_to_drop)
    elif method == 'mean':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif method == 'median':
        for col in columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif method == 'mode':
        for col in columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    elif method == 'constant':
        if fill_value is not None:
             for col in columns:
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def filter_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Filters outliers from a numeric column.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df
        
    df_clean = df.copy()
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    elif method == 'zscore':
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        if std == 0:
            return df_clean
        z_scores = (df_clean[column] - mean) / std
        df_clean = df_clean[abs(z_scores) < threshold] # threshold usually 3
        
    return df_clean

def scale_features(df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
    """
    Scales numeric features.
    """
    df_scaled = df.copy()
    if not columns:
        return df_scaled
        
    if method == 'standard':
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        
    return df_scaled

def encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
    """
    Encodes categorical features.
    """
    df_encoded = df.copy()
    if not columns:
        return df_encoded

    if method == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=False)
    elif method == 'label':
        le = LabelEncoder()
        for col in columns:
            # Convert to string to handle mixed types safely before encoding
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
    return df_encoded

def convert_dtypes(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """
    Converts column data type.
    """
    df_converted = df.copy()
    try:
        if dtype == 'datetime':
            df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
        elif dtype == 'numeric':
            df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
        elif dtype == 'string':
            df_converted[column] = df_converted[column].astype(str)
        elif dtype == 'category':
            df_converted[column] = df_converted[column].astype('category')
    except Exception as e:
        logger.error(f"Failed to convert {column} to {dtype}: {e}")
        
    return df_converted
