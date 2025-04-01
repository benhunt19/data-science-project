from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.globals import DATA_FOLDER
import warnings

from src.models.modelClassBase import Model

warnings.simplefilter("ignore", UserWarning)

class DecisionTreeRegressionModel(Model):
    """
    Description:
        Basic wrapper around sklearn KNN Regressor class. Procudes a regression based on nearest neighbours
    Parameters:
        maxDepth (int): Maxium depth of questioning
        minSampleSplit (int): Minimum number of times to split an internal node
    """
    def __init__(self, maxDepth : int = 3, minSamplesSplit : int = 5):
        super().__init__()
        # Define model to be DecisionTreeRegressor from sklearn
        self.model = DecisionTreeRegressor(max_depth=maxDepth, min_samples_split=minSamplesSplit)
        
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame) -> None:
        self.model.fit(X=x_train, y=y_train)

    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        return self.model.predict(x_test)
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plot_tree(self.model, filled=True, feature_names=["Feature"])
        plt.show()
    
    @staticmethod
    def __model_name__():
        return 'DecisionTreeRegressionModel'