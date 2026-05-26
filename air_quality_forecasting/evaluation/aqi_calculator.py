"""
AQI calculation using CPCB standards.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

AQI_BREAKPOINTS = {
    'PM2.5': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (250, float('inf'), 401, 500)],
    'PM10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (430, float('inf'), 401, 500)],
    'NO2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (400, float('inf'), 401, 500)],
}

AQI_CATEGORIES = {(0, 50): 'Good', (51, 100): 'Satisfactory', (101, 200): 'Moderate',
                  (201, 300): 'Poor', (301, 400): 'Very Poor', (401, 500): 'Severe'}


class AQICalculator:
    """Calculate Air Quality Index using CPCB standards."""
    
    def __init__(self, breakpoints=AQI_BREAKPOINTS):
        self.breakpoints = breakpoints
    
    def get_subindex(self, concentration, pollutant):
        """Calculate AQI subindex for a pollutant."""
        if pollutant not in self.breakpoints:
            logger.warning(f"Pollutant {pollutant} not in breakpoints")
            return np.nan
        
        breakpoints = self.breakpoints[pollutant]
        
        for (cl, ch, il, ih) in breakpoints:
            if ch == float('inf'):
                if concentration >= cl:
                    return 500
            elif cl <= concentration <= ch:
                return ((ih - il) / (ch - cl)) * (concentration - cl) + il
        
        return np.nan
    
    def compute_aqi(self, row, pollutants):
        """Compute composite AQI from multiple pollutants."""
        subindices = []
        
        for pollutant in pollutants:
            if pollutant in row.index and not pd.isna(row[pollutant]):
                subindex = self.get_subindex(row[pollutant], pollutant)
                if not np.isnan(subindex):
                    subindices.append(subindex)
        
        return max(subindices) if subindices else np.nan
    
    def get_category(self, aqi):
        """Get AQI category from AQI value."""
        if pd.isna(aqi):
            return 'N/A'
        
        for (lower, upper), category in AQI_CATEGORIES.items():
            if lower <= aqi <= upper:
                return category
        
        return 'Severe' if aqi > 400 else 'N/A'
    
    def calculate_dataframe(self, data, pollutants):
        """Calculate AQI for entire dataframe."""
        data_copy = data.copy()
        data_copy['AQI'] = data_copy.apply(
            lambda row: self.compute_aqi(row, pollutants), axis=1
        )
        data_copy['AQI_Category'] = data_copy['AQI'].apply(self.get_category)
        logger.info(f"AQI calculated for {len(data_copy)} rows")
        return data_copy
