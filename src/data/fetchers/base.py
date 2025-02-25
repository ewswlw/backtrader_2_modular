"""Base configuration class for data fetchers."""
import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

class BaseDataFetcherConfig:
    """Base configuration class for data fetchers."""
    
    def __init__(self, config_path: str = None):
        """Initialize with config path."""
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = os.path.join(project_root, 'config', 'data_sources')
        
        self.config_path = config_path
        self.config = {}
        self.active_config = None
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML files."""
        self.all_configs = {'configurations': []}
        
        # Load all YAML files in the config directory
        try:
            for file in os.listdir(self.config_path):
                if file.endswith('.yaml'):
                    file_path = os.path.join(self.config_path, file)
                    logging.info(f"Loading config from {file_path}")
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                        if config and 'configurations' in config:
                            self.all_configs['configurations'].extend(config['configurations'])
            
            if not self.all_configs['configurations']:
                raise ValueError(f"No valid configurations found in {self.config_path}")
            
            # Set first config as active by default
            self.set_active_config(self.all_configs['configurations'][0])
            logging.info(f"Loaded {len(self.all_configs['configurations'])} configurations")
            
        except Exception as e:
            logging.error(f"Error loading configurations: {str(e)}")
            raise
    
    def set_active_config(self, config: Dict[str, Any]):
        """Set the active configuration."""
        self.config = config
        self.active_config = config.get('name')
        logging.info(f"Set active configuration: {self.active_config}")
    
    def get_all_configs(self) -> List[Dict[str, Any]]:
        """Get all available configurations."""
        return self.all_configs['configurations']
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Get default settings from active configuration."""
        return self.config.get('default_settings', {})
    
    def get_bad_data_points(self) -> List[Dict[str, Any]]:
        """Get bad data points from active configuration."""
        return self.config.get('data_cleaning', {}).get('bad_data_points', [])

    def get_mapping(self, config_name: str) -> Dict[Tuple[str, str], str]:
        """Get Bloomberg mapping for the active configuration.
        
        Args:
            config_name: Name of the configuration to get mapping for
            
        Returns:
            Dictionary mapping (security, field) tuples to column names
        """
        if not self.config:
            raise ValueError("No active configuration set")
            
        mapping = {}
        data_mappings = self.config.get('data_mappings', {})
        
        for section_name, section_data in data_mappings.items():
            for field_name, field_info in section_data.items():
                security = field_info.get('security')
                field = field_info.get('field')
                alias = field_info.get('alias', field_name)
                
                if security and field:
                    mapping[(security, field)] = alias
                else:
                    logging.warning(f"Incomplete mapping for {field_name} in {section_name}")
        
        return mapping
