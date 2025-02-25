"""Market data fetcher implementation."""
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from src.utils.bloomberg.bloomberg_utils import fetch_bloomberg_data
from src.utils.data.transformations import convert_er_ytd_to_index
from src.utils.data.merger import merge_dfs
from src.data.fetchers.base import BaseDataFetcherConfig

class MarketDataFetcher:
    """Fetches market data based on configuration."""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config path."""
        try:
            self.config_handler = BaseDataFetcherConfig(config_path)
        except Exception as e:
            logging.error(f"Failed to initialize config handler: {str(e)}")
            raise
    
    def create_bloomberg_mapping(self, section: str) -> Dict[Tuple[str, str], str]:
        """Create Bloomberg API mapping from configuration section."""
        mapping = {}
        try:
            section_data = self.config_handler.config.get('data_mappings', {}).get(section, {})
            
            for ticker, info in section_data.items():
                # Create tuple key (security, field) and map to alias
                mapping[(ticker, info['field'])] = info['alias']
            
            logging.debug(f"Created mapping for section {section} with {len(mapping)} items")
            return mapping
            
        except Exception as e:
            logging.error(f"Error creating Bloomberg mapping for section {section}: {str(e)}")
            raise
    
    def fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data based on configuration.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping configuration names to their respective DataFrames
        """
        results = {}
        config = BaseDataFetcherConfig()
        
        for config_data in config.all_configs['configurations']:
            config_name = config_data.get('name', 'unnamed_config')
            logging.info(f"Processing configuration: {config_name}")
            
            # Get mapping for this configuration
            config.set_active_config(config_data)
            mapping = config.get_mapping(config_name)
            if not mapping:
                logging.warning(f"No mapping found for configuration: {config_name}")
                continue
            
            # Get dates from config
            default_settings = config_data.get('default_settings', {})
            start_date = default_settings.get('start_date')
            end_date = default_settings.get('end_date')  # Will be None if not specified
            periodicity = default_settings.get('periodicity', 'D')
            
            if not start_date:
                logging.warning(f"No start_date specified for {config_name}, skipping")
                continue
                
            logging.info(f"Fetching data for {config_name} from {start_date} to {end_date or 'now'}")
                
            # Fetch data using Bloomberg API
            try:
                df = fetch_bloomberg_data(
                    mapping=mapping,
                    start_date=start_date,
                    end_date=end_date,
                    periodicity=periodicity
                )
                
                # Store the result
                results[config_name] = df
                
                # Save to CSV if output path is specified
                if output_config := config_data.get('output'):
                    filename = output_config.get('filename')
                    directory = output_config.get('directory', 'data')
                    
                    # Create full output path
                    output_dir = os.path.join(os.path.dirname(config.config_path), '..', directory)
                    output_dir = os.path.abspath(output_dir)
                    output_path = os.path.join(output_dir, filename)
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save data
                    df.to_csv(output_path)
                    logging.info(f"Saved data to {output_path}")
                    
            except Exception as e:
                logging.error(f"Error fetching data for {config_name}: {str(e)}")
                continue
        
        return results
    
    def save_data(self, data: pd.DataFrame) -> str:
        """Save the data to CSV based on config settings."""
        try:
            output_config = self.config_handler.config.get('output', {})
            directory = output_config.get('directory', 'data/processed')
            filename = output_config.get('filename', 'data.csv')
            
            # Convert relative path to absolute path
            if not os.path.isabs(directory):
                project_root = Path(__file__).parent.parent.parent.parent
                directory = os.path.join(project_root, directory)
            
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save to CSV
            output_path = os.path.join(directory, filename)
            data.to_csv(output_path)
            logging.info(f"Saved data to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")
            raise
    
    def process_all_configurations(self):
        """Process all configurations and save data to respective CSVs."""
        results = {}
        
        for config in self.config_handler.get_all_configs():
            config_name = config['name']
            description = config.get('description', 'No description provided')
            logging.info(f"\nProcessing configuration: {config_name}")
            logging.info(f"Description: {description}")
            
            # Set the active configuration
            self.config_handler.set_active_config(config)
            
            # Fetch and save data
            try:
                data = self.fetch_market_data()
                output_file = self.save_data(data)
                results[config_name] = data
                
                logging.info(f"Successfully processed {config_name}")
                logging.info(f"Data saved to: {output_file}")
                logging.info(f"Shape: {data.shape}")
                logging.info(f"Columns: {', '.join(data.columns)}")
                
            except Exception as e:
                logging.error(f"Error processing {config_name}: {str(e)}")
                logging.error("Continuing with next configuration...")
                continue
        
        return results
