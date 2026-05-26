"""
Data loader for CPCB air quality datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and parse CPCB air quality data."""
    
    def __init__(self, csv_path, date_col='Date', time_col=None, location_col='City'):
        self.csv_path = csv_path
        self.date_col = date_col
        self.time_col = time_col
        self.location_col = location_col
        self.data = None
        
    def load(self):
        """Load data from CSV file."""
        try:
            self.data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded data from {self.csv_path}")
            logger.info(f"Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def parse_dates(self):
        """Parse date columns."""
        if self.time_col and self.time_col in self.data.columns:
            self.data['_datetime'] = pd.to_datetime(
                self.data[self.date_col] + ' ' + self.data[self.time_col]
            )
        else:
            self.data['_datetime'] = pd.to_datetime(
                self.data[self.date_col], errors='coerce'
            )
        logger.info(f"Date range: {self.data['_datetime'].min()} to {self.data['_datetime'].max()}")
        return self.data
    
    def get_locations(self):
        """Get unique locations in dataset."""
        if self.location_col not in self.data.columns:
            logger.warning("Location column not found")
            return []
        
        locations = self.data[self.location_col].unique()
        logger.info(f"Found {len(locations)} locations: {list(locations)}")
        return list(locations)
    
    def get_city_data(self, city):
        """Get data for a specific city."""
        if self.data is None:
            self.load()
        
        city_data = self.data[self.data[self.location_col] == city].copy()
        city_data = city_data.sort_values('_datetime').reset_index(drop=True)
        logger.info(f"City '{city}' has {len(city_data)} records")
        return city_data
    
    def get_statistics(self):
        """Get basic statistics about the dataset."""
        if self.data is None:
            self.load()
        
        stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'locations': self.get_locations(),
            'date_range': (self.data['_datetime'].min(), self.data['_datetime'].max()),
        }
        return stats
