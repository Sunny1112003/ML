"""
Fault injection framework for testing model robustness.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FaultInjector:
    """Inject faults into data to test model robustness."""
    
    def __init__(self, pollutant_cols=None):
        if pollutant_cols is None:
            pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']
        self.pollutant_cols = pollutant_cols
    
    def inject_missing_data(self, data, missing_rate=0.2):
        """Inject missing data (NaN values)."""
        data_faulty = data.copy()
        
        for col in self.pollutant_cols:
            if col not in data_faulty.columns:
                continue
            mask = np.random.random(len(data_faulty)) < missing_rate
            data_faulty.loc[mask, col] = np.nan
        
        # Forward fill then backward fill
        for col in self.pollutant_cols:
            if col in data_faulty.columns:
                data_faulty[col] = data_faulty[col].fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Injected missing data at rate {missing_rate*100:.1f}%")
        return data_faulty
    
    def inject_bias(self, data, bias_value=5.0):
        """Inject constant bias (systematic error)."""
        data_faulty = data.copy()
        
        for col in self.pollutant_cols:
            if col in data_faulty.columns:
                data_faulty[col] = data_faulty[col] + bias_value
        
        logger.info(f"Injected bias of {bias_value}")
        return data_faulty
    
    def inject_drift(self, data, drift_rate=0.01):
        """Inject drift (gradual degradation)."""
        data_faulty = data.copy()
        n_samples = len(data_faulty)
        
        for col in self.pollutant_cols:
            if col in data_faulty.columns:
                drift_values = np.linspace(0, drift_rate * data_faulty[col].mean(), n_samples)
                data_faulty[col] = data_faulty[col] + drift_values
        
        logger.info(f"Injected drift at rate {drift_rate*100:.1f}%")
        return data_faulty
    
    def inject_spikes(self, data, spike_rate=0.05, spike_magnitude=50.0):
        """Inject sudden spikes (anomalies)."""
        data_faulty = data.copy()
        
        for col in self.pollutant_cols:
            if col not in data_faulty.columns:
                continue
            spike_indices = np.random.random(len(data_faulty)) < spike_rate
            spike_values = np.random.normal(spike_magnitude, spike_magnitude*0.1, sum(spike_indices))
            data_faulty.loc[spike_indices, col] = spike_values
        
        logger.info(f"Injected spikes at rate {spike_rate*100:.1f}%")
        return data_faulty
    
    def inject_combined(self, data, missing_rate=0.1, bias=2.0, drift_rate=0.005, spike_rate=0.02):
        """Inject multiple faults combined."""
        data_faulty = data.copy()
        data_faulty = self.inject_missing_data(data_faulty, missing_rate)
        data_faulty = self.inject_bias(data_faulty, bias)
        data_faulty = self.inject_drift(data_faulty, drift_rate)
        data_faulty = self.inject_spikes(data_faulty, spike_rate)
        logger.info("Injected combined faults")
        return data_faulty
