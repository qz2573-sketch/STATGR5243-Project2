import io
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from src.utils import logger

def load_dataset(file_path: str, file_type: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Loads a dataset from a file path.
    
    Args:
        file_path (str): Path to the file.
        file_type (str, optional): Type of the file (csv, xlsx, json, parquet). 
                                   If None, inferred from extension.
    
    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: The loaded DataFrame and an error message if any.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None, f"File not found: {file_path}"
        
        if file_type is None:
            suffix = path.suffix.lower()
            if suffix == '.csv':
                file_type = 'csv'
            elif suffix in ['.xlsx', '.xls']:
                file_type = 'xlsx'
            elif suffix == '.json':
                file_type = 'json'
            elif suffix == '.parquet':
                file_type = 'parquet'
            else:
                return None, f"Unsupported file extension: {suffix}"

        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'xlsx':
            df = pd.read_excel(file_path)
        elif file_type == 'json':
            df = pd.read_json(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            return None, f"Unsupported file type: {file_type}"
            
        logger.info(f"Successfully loaded {file_path} with shape {df.shape}")
        return df, None

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None, str(e)

def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Returns summary information about the dataframe.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum()
    }
