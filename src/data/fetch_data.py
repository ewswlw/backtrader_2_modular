"""Main script to run data fetching."""
import os
import sys
import logging
from pathlib import Path
from typing import Dict

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from src.data.fetchers.market_data_fetcher import MarketDataFetcher

def print_dataframe_info(df: pd.DataFrame, name: str) -> None:
    """Print formatted information about a DataFrame."""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"{'='*80}")
    
    if df is None or df.empty:
        print(f"No data available for {name}")
        return
    
    print("\nDataFrame Info:")
    print("-" * 40)
    df.info()
    
    print("\nFirst 5 rows:")
    print("-" * 40)
    print(df.head())
    
    print("\nLast 5 rows:")
    print("-" * 40)
    print(df.tail())
    
    print("\nDescriptive Statistics:")
    print("-" * 40)
    print(df.describe())

def main():
    """Run data fetching process."""
    logging.info("Starting data fetching process...")
    
    # Initialize and run market data fetcher
    market_fetcher = MarketDataFetcher()
    market_data = market_fetcher.fetch_market_data()
    
    if not market_data:
        logging.warning("No data was fetched!")
        return
        
    # Print information for each DataFrame
    logging.info(f"Fetched data for {len(market_data)} configurations")
    for name, df in market_data.items():
        print_dataframe_info(df, name)
    
    logging.info("Successfully completed data fetching process")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
