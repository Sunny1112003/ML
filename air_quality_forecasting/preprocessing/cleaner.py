"""
Data cleaning utilities for handling missing values and outliers.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess air quality data."""
    
    def __init__(self, strategy='forward_fill'):
        self.strategy = strategy
    
    def clean(self, data, location_col='City', pollutant_cols=None):
        """Clean data by handling missing values."""
        if pollutant_cols is None:
            pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
        
        cleaned_list = []
        
        for location, group in data.groupby(location_col):
            group = group.sort_values('_datetime').copy()
            
            # Apply cleaning strategy
            for col in pollutant_cols:
                if col not in group.columns:
                    continue
                    
                if self.strategy == 'forward_fill':
                    group[col] = group[col].fillna(method='ffill').fillna(method='bfill')
                elif self.strategy == 'backward_fill':
                    group[col] = group[col].fillna(method='bfill').fillna(method='ffill')
                elif self.strategy == 'median':
                    group[col] = group[col].fillna(group[col].median())
                
                # Fill remaining NaNs with global median
                if group[col].isnull().any():
                    global_median = data[col].median()
                    group[col] = group[col].fillna(global_median if not pd.isna(global_median) else 0)
            
            cleaned_list.append(group)
        
        data_clean = pd.concat(cleaned_list).reset_index(drop=True)
        
        # Verify no NaNs remain
        remaining_nans = data_clean[pollutant_cols].isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Warning: {remaining_nans} NaN values still present. Filling with 0.")
            data_clean[pollutant_cols] = data_clean[pollutant_cols].fillna(0)
        
        logger.info(f"Data cleaning complete. Final shape: {data_clean.shape}")
        return data_clean
