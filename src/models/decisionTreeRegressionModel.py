from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Use DecisionTreeRegressor for partitioning and LinearRegression in each region

from sklearn.tree import plot_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.globals import DATA_FOLDER, WELLS_MERGED_US_CANADA, WELLS_MERGED
import warnings

from src.models.modelClassBase import Model
from src.utils import WorldMap

warnings.simplefilter("ignore", UserWarning)

class DecisionTreeRegressionModel(Model):
    """
    Description:
        Basic wrapper around sklearn KNN Regressor class. Procudes a regression based on nearest neighbours
    Parameters:
        maxDepth (int): Maxium depth of questioning
        minSampleSplit (int): Minimum number of times to split an internal node
        model_type (str): 'mean' for the mean of the options once split, or 'linearRegression' for 
        criterion (str literal): squared_error, friedman_mse, absolute_error, poisson
    """
    def __init__(self, maxDepth: int = 3, minSamplesSplit: int = 5, criterion : str ='absolute_error'):
        super().__init__()
        
        self.maxDepth = maxDepth

        # Define model to be DecisionTreeRegressor from sklearn
        self.model = DecisionTreeRegressor(max_depth=maxDepth, min_samples_split=minSamplesSplit, criterion=criterion)
        
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.model.fit(X=x_train, y=y_train)

    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        return self.model.predict(x_test)
    
    def plot(self):
        # Set up the figure
        plt.figure(figsize=(12, 8))  # Increased size for readability
        
        # Plot the decision tree
        plot_tree(
            self.model,
            feature_names=['Latitude', 'Longitude'],  # Replace with your actual feature names
            class_names=self.model.classes_.astype(str).tolist() if hasattr(self.model, 'classes_') else None,  # For classification
            filled=True,  # Color nodes by class/value
            rounded=True,  # Rounded boxes for better aesthetics
            proportion=True,  # Show proportions instead of raw counts
            precision=2,  # Limit decimal places
            fontsize=10  # Adjust font size for readability
        )
        
        # Customize the plot
        plt.title("Decision Tree Visualization", pad=20, fontsize=14)
        plt.tight_layout()  # Adjust layout to prevent overlap
        
        # Optional: Add a legend (manual, as plot_tree doesn't provide one)
        if hasattr(self.model, 'classes_'):
            legend_labels = [f"Class {cls}" for cls in self.model.classes_]
            plt.legend(legend_labels, title="Classes", loc='upper right')
        
        plt.show()


    def plotSplits(self, lat_lim=None, long_lim=None):
        # Create a grid of points covering the map
        lon_grid = np.linspace(-180, 180, 1000)
        lat_grid = np.linspace(-90, 90, 1000)
        xx, yy = np.meshgrid(lon_grid, lat_grid)
        
        # Reshape for prediction
        grid_points = np.column_stack((yy.ravel(), xx.ravel()))
        
        # Get predictions for all points
        predictions = self.model.predict(grid_points)
        
        # Create figure and axis BEFORE plotting
        fig, ax = plt.subplots(figsize=(15, 10))  # Creates `fig` and `ax` correctly
        
        # Plot decision boundaries first (background)
        sc = ax.scatter(xx.ravel(), yy.ravel(), c=predictions, cmap='viridis', alpha=0.5)
        
        # Overlay world map
        wm = WorldMap.worldMapPlot()
        wm.plot(ax=ax, linewidth=1, color='black')  # Ensure `plot` method uses `ax`

        # Add colorbar to figure
        cbar = fig.colorbar(sc, ax=ax, label='Predicted TVD')  
        
        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Average Depth Prediction for Each Split in North America (Tree Depth = {self.maxDepth})')
        ax.grid(True)
        
        # Apply limits if specified
        if lat_lim is not None:
            ax.set_ylim(lat_lim)
            
        if long_lim is not None:
            ax.set_xlim(long_lim)
        
        plt.show()

    
    @staticmethod
    def __model_name__():
        return 'DecisionTreeRegressionModel'

if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 10_000
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    kwargs = {
        'maxDepth': 20,
        'minSamplesSplit': 10,
    }
    
    dtr = DecisionTreeRegressionModel(**kwargs)
    
    dtr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    dtr.test(
        x_test=test_df[['lat', 'lon',]],
        y_test=test_df['tvd']
    )
    
    plot_kwargs = {
        'lat_lim': [0, 80],
        'long_lim': [-150, -60]
    }
    dtr.plotSplits(**plot_kwargs)
    
    # dtr.plot()
    # dtr.worldMapPlot()
    