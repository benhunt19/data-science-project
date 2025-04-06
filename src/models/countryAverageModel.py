from src.models.modelClassBase import Model
import pandas as pd
import numpy as np

class CountryAverageModel(Model):
    def __init__(self):
        """Initialize the CountryAverageModel with empty attributes."""
        super().__init__()
        self.combiDF = None
        self.country_averages = None  # Renamed for clarity
    
    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train the model by computing country-specific averages from the training data.
        
        Args:
            x_train (pd.DataFrame): Training features with a 'country' column.
            y_train (pd.DataFrame): Training target values.
        """
        # Combine features and target into a single DataFrame
        self.combiDF = x_train.copy()
        self.combiDF['y_train'] = y_train
        
        # Compute mean target value per country
        self.country_averages = self.combiDF.groupby('country')['y_train'].mean()
    
    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame = None) -> np.ndarray:
        """
        Predict target values for the test set based on country averages.
        
        Args:
            x_test (pd.DataFrame): Test features with a 'country' column.
            y_test (pd.DataFrame, optional): Test target values (unused, included for API consistency).
        
        Returns:
            np.ndarray: Predicted values for each test sample.
        """
        # Vectorized prediction using map, with fallback to overall mean
        predictions = x_test['country'].map(self.country_averages).fillna(self.country_averages.mean())
        return predictions.to_numpy()