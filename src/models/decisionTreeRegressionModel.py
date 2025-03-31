from sklearn.tree import DecisionTreeRegressor

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
    """
    def __init__(self):
        super().__init__()
        # Define model to be DecisionTreeRegressor from sklearn
        self.model = DecisionTreeRegressor()
        
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame):
        self.model.fit(X=x_train, y=y_train)

    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        return self.model.predict(x_test)