"""
Central configuration for Air Quality Forecasting project.
"""

import os
from datetime import datetime

# ============== DATA CONFIGURATION ==============
DATA_PATH = 'data/city_day.csv'
DATE_COL = 'Date'
TIME_COL = None  # Set to column name if separate time column exists
LOCATION_COL = 'City'

# Pollutant columns to model
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NH3']

# Cities in dataset (26 Indian cities)
CITIES = [
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
    'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
    'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata',
    'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
    'Visakhapatnam'
]

# ============== MODEL CONFIGURATION ==============
SEQUENCE_LENGTH = 30  # Look-back window (30 days)
FORECAST_STEPS = 7   # Predict next 7 days
TRAIN_SPLIT = 0.7    # 70% train, 30% test
VALIDATION_SPLIT = 0.2

# ConvLSTM Architecture
CONVLSTM_FILTERS = [32, 64]
CONVLSTM_KERNEL = (3, 3)
DROPOUT_RATE = 0.2
DENSE_UNITS = 128
BATCH_NORMALIZATION = True

# ============== TRAINING CONFIGURATION ==============
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mse'
EARLY_STOPPING_PATIENCE = 10
VERBOSE = 1

# ============== PREPROCESSING CONFIGURATION ==============
MISSING_VALUE_STRATEGY = 'forward_fill'  # Options: 'forward_fill', 'backward_fill', 'median'
SCALER_TYPE = 'minmax'  # Options: 'minmax', 'standard'
OUTLIER_DETECTION = True
OUTLIER_THRESHOLD = 3.0  # Standard deviations

# ============== AQI CONFIGURATION ==============
# CPCB AQI Breakpoints
AQI_BREAKPOINTS = {
    'PM2.5': [
        (0, 30, 0, 50),          # Good
        (31, 60, 51, 100),       # Satisfactory
        (61, 90, 101, 200),      # Moderate
        (91, 120, 201, 300),     # Poor
        (121, 250, 301, 400),    # Very Poor
        (250, float('inf'), 401, 500)   # Severe
    ],
    'PM10': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (430, float('inf'), 401, 500)
    ],
    'NO2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (400, float('inf'), 401, 500)
    ],
    'SO2': [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1600, float('inf'), 401, 500)
    ],
    'CO': [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.0, float('inf'), 401, 500)
    ],
    'O3': [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (748, float('inf'), 401, 500)
    ],
    'NH3': [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1800, float('inf'), 401, 500)
    ]
}

AQI_CATEGORIES = {
    (0, 50): 'Good',
    (51, 100): 'Satisfactory',
    (101, 200): 'Moderate',
    (201, 300): 'Poor',
    (301, 400): 'Very Poor',
    (401, 500): 'Severe'
}

# ============== FAULT INJECTION CONFIGURATION ==============
FAULT_TYPES = ['missing_data', 'bias', 'drift', 'spikes']
FAULT_SEVERITY_LEVELS = [0.1, 0.2, 0.3]  # 10%, 20%, 30%

# Fault Parameters
MISSING_DATA_RATE = 0.2
BIAS_VALUE = 5.0
DRIFT_RATE = 0.01
SPIKE_RATE = 0.05
SPIKE_MAGNITUDE = 50.0

# ============== OUTPUT CONFIGURATION ==============
RESULTS_DIR = 'air_quality_forecasting/results/'
MODELS_DIR = 'air_quality_forecasting/results/models/'
PLOTS_DIR = 'air_quality_forecasting/results/plots/'
LOGS_DIR = 'air_quality_forecasting/results/logs/'

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============== LOGGING CONFIGURATION ==============
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, f'air_quality_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# ============== RANDOM SEED ==============
RANDOM_SEED = 42
