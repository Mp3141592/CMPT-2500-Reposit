
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import logging
import threading
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegressionMonitor():
    def __init__(self, port=8002):

        self.port = port

        
        # Regression-specific metrics
        self.mse = Gauge('regression_mean_squared_error', 'Mean Squared Error')
        self.rmse = Gauge('regression_root_mean_squared_error', 'Root Mean Squared Error')
        self.mae = Gauge('regression_mean_absolute_error', 'Mean Absolute Error')
        self.r_squared = Gauge('regression_r_squared', 'R-squared coefficient')
        
        # Feature importance tracking (top 5 features)
        self.feature_importance = Gauge('feature_importance', 'Feature importance value', ['feature_name'])

    def start(self):
        """Start the Prometheus HTTP server and resource monitoring thread"""
        try:
            # Start the Prometheus HTTP server
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start the resource monitoring thread
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring thread started")
            
            return True
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            return False

    def record_metrics(self, mse=None, rmse=None, mae=None, r_squared=None, feature_importance=None):
        
        """Record regression metrics"""
        if mse is not None:
            self.mse.set(mse)
        if rmse is not None:
            self.rmse.set(rmse)
        if mae is not None:
            self.mae.set(mae)
        if r_squared is not None:
            self.r_squared.set(r_squared)
            
        # Update feature importance for top features
        if feature_importance is not None:
            # Sort features by importance (assuming dict format)
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature_name, importance in sorted_features:
                self.feature_importance.labels(feature_name=feature_name).set(importance)

def get_training_monitor(port=8002):
    """Get or create the training monitor singleton instance"""
    if not hasattr(get_training_monitor, 'instance'):
        get_training_monitor.instance = RegressionMonitor(port=port)
    return get_training_monitor.instance