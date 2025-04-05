from src.models.modelClassBase import Model

# Well depeth K nearest neighbour regression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.globals import DATA_FOLDER
import warnings
from src.globals import WELLS_MERGED_US_CANADA

warnings.simplefilter("ignore", UserWarning)

class KNNRegressionModel(Model):
    """
    Description:
        Basic wrapper around sklearn KNN Regressor class. Procudes a regression based on nearest neighbours
    Parameters:
        k (int): Number of nearest neighbours to find before running the regression
        weights (str or function): Defauly is 'uniform', can be 'distance' to be inversly related to distance, can even be a single parameter function eg lambda x : x**" 
    """
    def __init__(self, k: int, weights: str = None):
        super().__init__()
        self.k = k
        # Allows None to be passed in as a Kwarg
        self.weights = 'uniform' if weights is None else weights
        self.model = KNeighborsRegressor(n_neighbors=self.k, weights=self.weights)
        
        
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame):
        self.model.fit(X=x_train, y=y_train)

    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        return self.model.predict(x_test)
    
    @staticmethod
    def __model_name__():
        return 'KNNRegressionModel'

if __name__ == "__main__":
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged = wells_merged[wells_merged['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    
    sample_size = 100_000
    train_pcnt = 0.8
    
    df = wells_merged.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    kwargs = {
        'k': 10
    }
    
    knn = KNNRegressionModel(**kwargs)
    knn.train(
        x_train=train_df[['lat', 'lon']],
        y_train=train_df['tvd']
    )

    test_vals = knn.test(
        x_test=test_df[['lat', 'lon']],
        y_test=test_df['tvd']
    )

    # Calculate the difference between predicted and actual values
    diff = test_vals - test_df['tvd']
    
    # Create a histogram of the differences with KDE
    plt.figure(figsize=(10, 6))
    sns.set_style('darkgrid')
    bins = 80
    sns.histplot(diff, bins=bins, alpha=0.7, color='blue', edgecolor='black', kde=True)
    plt.title(f'Histogram of Prediction Errors for {kwargs["k"]} Nearest Neighbours, {bins} Bins')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.7, linewidth=1.5)  # Increased alpha and linewidth for thicker gridlines
    plt.tight_layout()
    plt.show()