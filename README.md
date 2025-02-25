# Modular Backtesting System

A robust, modular backtesting system for financial market analysis with integrated data fetching from Bloomberg.

## Project Structure

```
backtrader_2_modular/
├── config/
│   └── data_sources/         # Data source configurations
│       ├── credit_data.yaml  # Credit market data config
│       └── market_data.yaml  # Market indicators config
├── data/
│   ├── raw/                  # Raw data storage
│   └── processed/            # Processed data storage
│       └── daily/           # Daily frequency data
├── src/
│   ├── data/
│   │   ├── fetchers/        # Data fetching modules
│   │   └── fetch_data.py    # Main data fetching script
│   └── utils/
│       ├── bloomberg/       # Bloomberg API utilities
│       ├── data/           # Data processing utilities
│       └── metrics/        # Financial metrics calculations
└── tests/                   # Test suite
```

## Features

### Data Fetching
- Modular data fetching system with support for multiple data sources
- Bloomberg data integration via xbbg
- Configurable data cleaning and processing
- Support for various data frequencies (daily, weekly, monthly)

### Data Processing
- Forward-fill missing values
- Handle bad data points with configurable actions
- Date alignment across multiple series
- Excess return calculations and transformations

### Configuration System
- YAML-based configuration for easy maintenance
- Separate configs for different data types (market data, credit data)
- Flexible mapping system for Bloomberg fields

## Setup and Installation

1. Install dependencies:
```bash
poetry install
```

2. Configure data sources:
- Edit YAML files in `config/data_sources/`
- Set Bloomberg tickers and field mappings
- Configure data cleaning rules

## Usage

### Data Fetching

Run the main data fetching script:
```bash
poetry run python src/data/fetch_data.py
```

### Configuration Format

Example market data configuration:
```yaml
configurations:
  - name: daily_market_and_credit_data
    description: Daily frequency data including credit spreads and market indices
    default_settings:
      start_date: '2002-10-01'
      periodicity: 'D'
      align_start: true
    data_mappings:
      market_data:
        TICKER Index:
          security: TICKER Index
          field: PX_LAST
          alias: friendly_name
          description: Description
    data_cleaning:
      fill_method: ffill
      bad_data_points:
        - field: field_name
          start_date: 'YYYY-MM-DD'
          end_date: 'YYYY-MM-DD'
          action: remove
    output:
      filename: output_file.csv
      directory: data/processed/daily
```

### Data Fields

#### Market Data
- Credit Spreads (OAS)
  - CAD Investment Grade
  - US High Yield
  - US Investment Grade
- Market Indices
  - TSX
  - VIX
- Interest Rates
  - US 3M-30Y Slope
- Economic Indicators
  - US Growth Surprises
  - US Inflation Surprises
  - US LEI YoY
  - US Hard Data Surprises
  - US Economic Regime

## Data Quality

The system includes several data quality features:
1. Forward-fill missing values
2. Handle known bad data periods
3. Align data start dates
4. Log missing value percentages

## Utilities

### Data Processing
- `merge_dfs`: Merge multiple DataFrames with date alignment
- `convert_er_ytd_to_index`: Convert YTD excess returns to indices
- `get_ohlc`: Get OHLC data at various frequencies

### Bloomberg Utils
- `fetch_bloomberg_data`: Fetch data from Bloomberg with error handling
- Field mapping and data cleaning

## Contributing

1. Follow the project structure
2. Add tests for new features
3. Update configuration files as needed
4. Document changes in the README

## Dependencies

- Python 3.8+
- xbbg: Bloomberg data fetching
- pandas: Data manipulation
- PyYAML: Configuration management
- poetry: Dependency management

## License

[Your License Here]
