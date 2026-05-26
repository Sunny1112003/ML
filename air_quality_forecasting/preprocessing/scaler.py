"""
Scaling utilities for normalization.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataScaler:
    """Scale data for model training."""
    
    def __init__(self, scaler_type='minmax'):
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.is_fitted = False
    
    def fit(self, data, columns=None):
        """Fit scaler on training data."""
        if columns is None:
            columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
        
        self.scaler.fit(data[columns])
        self.is_fitted = True
        logger.info(f"Scaler fitted on {len(columns)} columns")
        return self
    
    def transform(self, data, columns=None):
        """Transform data using fitted scaler."""
        if columns is None:
            columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
            
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        data_scaled = data.copy()
        data_scaled[columns] = self.scaler.transform(data[columns])
        return data_scaled
    
    def inverse_transform(self, data, columns=None):
        """Inverse transform scaled data."""
        if columns is None:
            columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
            
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        data_unscaled = data.copy()
        data_unscaled[columns] = self.scaler.inverse_transform(data[columns])
        return data_unscaled
    
    def fit_transform(self, data, columns=None):
        """Fit and transform in one step."""
        if columns is None:
            columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
            
        self.fit(data, columns)
        return self.transform(data, columns)
