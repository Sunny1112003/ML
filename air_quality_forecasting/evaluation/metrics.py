"""
Evaluation metrics for air quality forecasting.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """Calculate evaluation metrics."""
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true, y_pred):
        """R² Score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def evaluate(y_true, y_pred):
        """Calculate all metrics."""
        metrics_dict = {
            'MAE': Metrics.mae(y_true, y_pred),
            'RMSE': Metrics.rmse(y_true, y_pred),
            'MSE': Metrics.mse(y_true, y_pred),
            'R2': Metrics.r2(y_true, y_pred),
            'MAPE': Metrics.mape(y_true, y_pred)
        }
        
        logger.info("Evaluation Metrics:")
        for metric, value in metrics_dict.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics_dict
