"""Bloomberg data fetching utilities."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, List, Union
from xbbg import blp
from datetime import datetime
from ..data.merger import merge_dfs
from ..data.transformations import convert_er_ytd_to_index

def fetch_bloomberg_data(
    mapping: Dict[Tuple[str, str], str],
    start_date: str,
    end_date: str = None,
    periodicity: str = 'D',
    align_start: bool = True
) -> pd.DataFrame:
    """
    Fetch data from Bloomberg using xbbg
    
    Args:
        mapping: Dictionary mapping (security, field) tuples to column names
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format, defaults to today
        periodicity: Data frequency ('D' for daily)
        align_start: Whether to align data from the start date
        
    Returns:
        DataFrame with requested data and datetime index
    """
    logging.info(f"Fetching Bloomberg data for {len(mapping)} fields")
    logging.debug(f"Mapping: {mapping}")
    
    securities = list(set(security for security, _ in mapping.keys()))
    fields = list(set(field for _, field in mapping.keys()))
    
    logging.info(f"Unique securities: {len(securities)}")
    logging.info(f"Unique fields: {len(fields)}")

    # Handle dates
    try:
        start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
    except Exception as e:
        logging.error(f"Error parsing dates: {str(e)}")
        raise ValueError(f"Invalid date format. Dates should be YYYY-MM-DD. Got start_date={start_date}, end_date={end_date}")
    
    logging.info(f"Date range: {start_date} to {end_date}")

    try:
        # Fetch data using xbbg
        logging.info("Calling Bloomberg API...")
        df = blp.bdh(
            tickers=securities,
            flds=fields,
            start_date=start_date,
            end_date=end_date,
            Per=periodicity
        )
        logging.info(f"Received data with shape: {df.shape}")

        # Create individual DataFrames for each security/field pair
        dfs_to_merge = []
        for (security, field), new_name in mapping.items():
            if (security, field) in df.columns:
                series = df[(security, field)]
                series.name = new_name
                dfs_to_merge.append(series)
            else:
                logging.warning(f"Missing data for {security} {field}")

        # Merge all DataFrames with proper alignment
        if dfs_to_merge:
            merged_df = merge_dfs(
                *dfs_to_merge,
                fill='ffill',  # Forward fill missing values
                start_date_align='yes' if align_start else 'no'  # Align start dates if requested
            )
            
            # Convert excess returns to index if any columns contain YTD excess returns
            if any('INDEX_EXCESS_RETURN_YTD' in field for _, field in mapping.keys()):
                merged_df = convert_er_ytd_to_index(merged_df)
        else:
            merged_df = pd.DataFrame()

        # Log data quality metrics
        missing_pct = (merged_df.isnull().sum() / len(merged_df)) * 100
        for col in merged_df.columns:
            if missing_pct[col] > 0:
                logging.warning(f"Column {col} has {missing_pct[col]:.2f}% missing values")

        return merged_df

    except Exception as e:
        logging.error(f"Error fetching Bloomberg data: {str(e)}")
        raise
