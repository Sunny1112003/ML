"""
Degradation testing framework for evaluating model robustness under faults.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from fault_injection.fault_injector import FaultInjector
from evaluation.metrics import Metrics

logger = logging.getLogger(__name__)


class DegradationTester:
    """Test model performance degradation under various faults."""
    
    def __init__(self, model, original_data, faulty_data, scaler=None):
        self.model = model
        self.original_data = original_data
        self.faulty_data = faulty_data
        self.scaler = scaler
        self.results = {}
    
    def test_fault_impact(self, fault_type, severity_levels, X_test, y_test):
        """Test impact of specific fault type."""
        injector = FaultInjector()
        results = {}
        
        for severity in severity_levels:
            # Inject fault
            if fault_type == 'missing_data':
                X_faulty = injector.inject_missing_data(X_test, missing_rate=severity)
            elif fault_type == 'bias':
                X_faulty = injector.inject_bias(X_test, bias_value=severity)
            elif fault_type == 'drift':
                X_faulty = injector.inject_drift(X_test, drift_rate=severity)
            elif fault_type == 'spikes':
                X_faulty = injector.inject_spikes(X_test, spike_rate=severity)
            else:
                logger.warning(f"Unknown fault type: {fault_type}")
                continue
            
            # Make predictions
            y_pred = self.model.predict(X_faulty)
            
            # Calculate metrics
            metrics = Metrics.evaluate(y_test, y_pred)
            results[severity] = metrics
        
        self.results[fault_type] = results
        return results
    
    def generate_report(self):
        """Generate degradation test report."""
        report = "\n" + "="*80 + "\n"
        report += "DEGRADATION TEST REPORT\n"
        report += "="*80 + "\n\n"
        
        for fault_type, results in self.results.items():
            report += f"Fault Type: {fault_type.replace('_', ' ').upper()}\n"
            report += "-" * 80 + "\n"
            
            for severity, metrics in sorted(results.items()):
                report += f"  Severity: {severity} -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}\n"
        
        report += "\n" + "="*80 + "\n"
        logger.info(report)
        return report
