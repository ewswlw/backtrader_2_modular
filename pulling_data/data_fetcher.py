import logging
from datetime import datetime
import pandas as pd
import os
import sys
import yaml
from typing import Dict, Tuple, Optional, Union
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.bloomberg_fetcher import fetch_bloomberg_data
from src.utils.transformations import convert_er_ytd_to_index
from src.utils.data_merger import merge_dfs

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(project_root, 'logs', 'data_fetcher.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class DataFetcherConfig:
    """Configuration class for DataFetcher"""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(project_root, 'config', 'market_data_config.yaml')
        self.logger = logging.getLogger(f"{__name__}.DataFetcherConfig")
        self.logger.info(f"Initializing DataFetcherConfig with path: {self.config_path}")
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        try:
            self.logger.debug(f"Attempting to load config from: {self.config_path}")
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Successfully loaded config with {len(config)} top-level keys")
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {self.config_path}. Using default mappings.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config: {str(e)}")
            raise
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        self.logger.info("Using default configuration")
        return {
            'market_data_mapping': {
                'I05510CA Index': {'INDEX_OAS_TSY_BP': 'cad_oas'},
                'LF98TRUU Index': {'INDEX_OAS_TSY_BP': 'us_hy_oas'},
                'LUACTRUU Index': {'INDEX_OAS_TSY_BP': 'us_ig_oas'},
                'SPTSX Index': {'PX_LAST': 'tsx'},
                'VIX Index': {'PX_LAST': 'vix'},
                'USYC3M30 Index': {'PX_LAST': 'us_3m_10y'},
                'BCMPUSGR Index': {'PX_LAST': 'us_growth_surprises'},
                'BCMPUSIF Index': {'PX_LAST': 'us_inflation_surprises'},
                'LEI YOY  Index': {'PX_LAST': 'us_lei_yoy'},
                '.HARDATA G Index': {'PX_LAST': 'us_hard_data_surprises'},
                '.ECONREGI G Index': {'PX_LAST': 'us_economic_regime'},
            },
            'er_ytd_mapping': {
                'I05510CA Index': {'INDEX_EXCESS_RETURN_YTD': 'cad_ig_er'},
                'LF98TRUU Index': {'INDEX_EXCESS_RETURN_YTD': 'us_hy_er'},
                'LUACTRUU Index': {'INDEX_EXCESS_RETURN_YTD': 'us_ig_er'},
            }
        }

class DataFetcher:
    """Class for fetching and processing market data"""
    def __init__(self, config: Optional[DataFetcherConfig] = None):
        self.config = config or DataFetcherConfig()
        self.logger = logging.getLogger(f"{__name__}.DataFetcher")
        self.logger.info("Initializing DataFetcher")
    
    def _validate_dates(self, start_date: str, end_date: Optional[str] = None) -> Tuple[str, str]:
        """Validate and process date inputs"""
        self.logger.debug(f"Validating dates - start: {start_date}, end: {end_date}")
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            if end_date:
                datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date = datetime.now().strftime('%Y-%m-%d')
                self.logger.debug(f"No end date provided, using current date: {end_date}")
            return start_date, end_date
        except ValueError as e:
            self.logger.error(f"Date validation failed: {str(e)}")
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}")

    def _prepare_mappings(self) -> Tuple[Dict, Dict]:
        """Convert config mappings to format expected by bloomberg_fetcher"""
        self.logger.debug("Preparing Bloomberg data mappings")
        try:
            market_mapping = {
                (ticker, field): col_name
                for ticker, fields in self.config.config['market_data_mapping'].items()
                for field, col_name in fields.items()
            }
            er_mapping = {
                (ticker, field): col_name
                for ticker, fields in self.config.config['er_ytd_mapping'].items()
                for field, col_name in fields.items()
            }
            self.logger.debug(f"Prepared {len(market_mapping)} market mappings and {len(er_mapping)} ER mappings")
            return market_mapping, er_mapping
        except KeyError as e:
            self.logger.error(f"Error preparing mappings: {str(e)}")
            raise KeyError(f"Missing required mapping in config: {str(e)}")

    def _validate_data_for_backtesting(self, df: pd.DataFrame) -> None:
        """Validate data quality specifically for backtesting purposes"""
        self.logger.info("Validating data for backtesting requirements")
        
        # Check index frequency
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq != 'M':
            self.logger.warning(f"Expected monthly frequency but got {inferred_freq}")
        
        # Check for outliers (using 3 standard deviations as threshold)
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[col][(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
            if not outliers.empty:
                self.logger.warning(f"Found {len(outliers)} outliers in {col}")
                self.logger.debug(f"Outliers in {col}:\n{outliers}")
        
        # Verify excess return indices are properly calculated
        er_cols = [col for col in df.columns if col.endswith('_er_index')]
        for col in er_cols:
            if df[col].iloc[0] != 100:  # Check if index starts at 100
                self.logger.warning(f"{col} does not start at 100")
            
            # Check for unrealistic returns
            returns = df[col].pct_change()
            extreme_returns = returns[abs(returns) > 0.2]  # 20% threshold
            if not extreme_returns.empty:
                self.logger.warning(f"Found {len(extreme_returns)} extreme returns in {col}")
                self.logger.debug(f"Extreme returns in {col}:\n{extreme_returns}")

    def _log_data_quality_metrics(self, df: pd.DataFrame) -> None:
        """Log comprehensive data quality metrics for debugging"""
        self.logger.info("Data Quality Metrics:")
        self.logger.info(f"Shape: {df.shape}")
        self.logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
        
        # Missing values analysis
        missing_vals = df.isnull().sum()
        if missing_vals.any():
            self.logger.warning("Missing Values Found:")
            self.logger.warning(f"\n{missing_vals[missing_vals > 0]}")
        
        # Basic statistics
        self.logger.info("\nBasic Statistics:")
        stats = df.describe()
        self.logger.info(f"\n{stats}")
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        self.logger.info(f"Memory Usage: {memory_usage:.2f} MB")
        
        # Index frequency analysis
        freq = pd.infer_freq(df.index)
        gaps = df.index.to_series().diff().value_counts()
        self.logger.info(f"Inferred Frequency: {freq}")
        if len(gaps) > 1:
            self.logger.warning("Irregular time gaps found:")
            self.logger.warning(f"\n{gaps}")

    def fetch_data(self, 
                  start_date: str = '2002-01-01',
                  end_date: Optional[str] = None,
                  output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch and process market data from Bloomberg
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional)
            output_path: Path to save the output CSV (optional)
            
        Returns:
            pd.DataFrame: Processed market data
        """
        try:
            # Validate dates
            start_date, end_date = self._validate_dates(start_date, end_date)
            self.logger.info(f"Fetching data from {start_date} to {end_date}")

            # Prepare mappings
            market_mapping, er_mapping = self._prepare_mappings()

            # Fetch market data
            self.logger.info("Fetching market data from Bloomberg")
            market_df = fetch_bloomberg_data(
                mapping=market_mapping,
                start_date=start_date,
                end_date=end_date,
                periodicity='M',  # Monthly data
                align_start=True
            )
            market_df = market_df.dropna()  # Remove any NaN values
            self.logger.debug(f"Received market data with shape: {market_df.shape}")
            self._log_data_quality_metrics(market_df)
            
            # Fetch excess return YTD data
            self.logger.info("Fetching excess return YTD data from Bloomberg")
            er_df = fetch_bloomberg_data(
                mapping=er_mapping,
                start_date=start_date,
                end_date=end_date,
                periodicity='M',  # Monthly data
                align_start=True
            )
            er_df = er_df.dropna()  # Remove any NaN values
            self.logger.debug(f"Received ER data with shape: {er_df.shape}")
            self._log_data_quality_metrics(er_df)

            # Convert excess returns to indices
            self.logger.info("Converting excess returns to indices")
            er_df.index = pd.to_datetime(er_df.index)  # Ensure datetime index
            er_index_df = convert_er_ytd_to_index(er_df)
            self.logger.debug(f"Created ER indices with shape: {er_index_df.shape}")
            self._log_data_quality_metrics(er_index_df)

            # Merge all dataframes
            self.logger.info("Merging all dataframes")
            final_df = merge_dfs(market_df, er_index_df, fill='ffill', start_date_align='yes')
            self.logger.info(f"Final dataset shape: {final_df.shape}")
            
            # Validate final dataset for backtesting
            self._validate_data_for_backtesting(final_df)
            self._log_data_quality_metrics(final_df)

            # Save to CSV if output path provided
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                final_df.to_csv(output_path)
                self.logger.info(f"Data saved to {output_path}")
            
            return final_df

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}", exc_info=True)
            raise

def fetch_market_data(start_date: str = '2002-01-01',
                     end_date: Optional[str] = None,
                     config_path: Optional[str] = None,
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to fetch market data using DataFetcher
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional)
        config_path: Path to config file (optional)
        output_path: Path to save the output CSV (optional)
        
    Returns:
        pd.DataFrame: Processed market data
    """
    logger = logging.getLogger(f"{__name__}.fetch_market_data")
    logger.info(f"Starting market data fetch with config_path: {config_path}")
    
    config = DataFetcherConfig(config_path) if config_path else None
    fetcher = DataFetcher(config)
    return fetcher.fetch_data(start_date, end_date, output_path)

if __name__ == '__main__':
    # Ensure logs directory exists
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    
    # Example usage
    try:
        logger.info("Starting main execution")
        output_path = os.path.join(project_root, 'pulling_data', 'backtest_data.csv')
        logger.info(f"Output path set to: {output_path}")
        
        data = fetch_market_data(output_path=output_path)
        
        print("\nDataset Info:")
        print(data.info())
        
        print("\nFirst few rows of the data:")
        print(data.head())
        
        logger.info("Main execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
