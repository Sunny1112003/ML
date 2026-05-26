"""
ConvLSTM model architecture for air quality forecasting.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)


class ConvLSTMModel:
    """ConvLSTM model for spatio-temporal air quality prediction."""
    
    def __init__(self, input_shape, output_size, 
                 filters=[32, 64],
                 kernel=(3, 3),
                 dropout=0.2,
                 dense_units=128,
                 use_batch_norm=True,
                 learning_rate=0.001):
        """
        Initialize ConvLSTM model.
        
        Args:
            input_shape: Tuple (timesteps, height, width, channels)
            output_size: Number of output features (e.g., 7 pollutants)
        """
        self.input_shape = input_shape
        self.output_size = output_size
        self.model = self._build_model(
            input_shape, output_size, filters, kernel, 
            dropout, dense_units, use_batch_norm
        )
        self.optimizer = Adam(learning_rate=learning_rate)
        self.compile_model()
    
    def _build_model(self, input_shape, output_size, filters, kernel, 
                    dropout, dense_units, use_batch_norm):
        """Build ConvLSTM model."""
        model = Sequential()
        
        # First ConvLSTM layer
        model.add(ConvLSTM2D(
            filters=filters[0],
            kernel_size=kernel,
            activation='relu',
            input_shape=input_shape,
            return_sequences=True
        ))
        model.add(Dropout(dropout))
        
        # Second ConvLSTM layer
        model.add(ConvLSTM2D(
            filters=filters[1],
            kernel_size=kernel,
            activation='relu',
            return_sequences=False
        ))
        model.add(Dropout(dropout))
        
        if use_batch_norm:
            model.add(BatchNormalization())
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(output_size, activation='linear'))
        
        logger.info(f"ConvLSTM model created with shape {input_shape}")
        return model
    
    def compile_model(self):
        """Compile model."""
        self.model.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=['mae']
        )
        logger.info("Model compiled")
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             epochs=50, batch_size=32, verbose=1):
        """Train the model."""
        val_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        logger.info(f"Model training complete. Final loss: {history.history['loss'][-1]:.4f}")
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save model to file."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
