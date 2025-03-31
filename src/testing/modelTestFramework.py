from src.models.knnRegressionModel import KNNRegressionModel
from src.models.nnModel import NNModel
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel

from src.globals import DATA_FOLDER

import pandas as pd
import numpy as np

import torch.nn as nn

class ModelTestFramework:
    def __init__(self):
        pass
    
    def testModel(self, modelMeta : list[dict], x_train: pd.DataFrame, y_train: pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame) -> None:
        # Create a new instance of the model
        m = modelMeta['model'](**modelMeta['kwargs'])
        
        # Train the model on the test data
        m.train(x_train=x_train, y_train=y_train)
        
        # Test the trained model on the testing set
        testRes = m.test(x_test=x_test, y_test=y_test)
        print(testRes)

    def testModels(self, modelMetas : list[dict], x_train: pd.DataFrame, y_train: pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame) -> None:
        """
        Description:
            Run testModel over an array of models with various parameters and kwargs
        Parameters:
            modalMetas (list[dict]): Array of meta data for each model to be tested
        """
        for modelMeta in modelMetas:
            self.testModel(
                modelMeta=modelMeta,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )
        
        return self

if __name__ == "__main__":
    
    modelMetas = [
        {
        'model': KNNRegressionModel,
        'kwargs': {
            'k': 5
        }
        },
        {
            'model': KNNRegressionModel,
            'kwargs': {
                'k': 5,
                'weights': 'distance'
            }
        },
        {
            'model': DecisionTreeRegressionModel,
            'kwargs': {}
        },
        {
            'model': NNModel,
            'kwargs': {
                'network_meta': [
                    {
                        'neurons': 2 # input layer / input dimension
                    },
                    {
                        'neurons': 30,
                        'activation': nn.ReLU,
                        'type': nn.Linear
                    },
                    {
                        'neurons': 50,
                        'activation': nn.ReLU,
                        'type': nn.Linear
                    },
                    {
                        'neurons': 50,
                        'activation': nn.ReLU,
                        'type': nn.Linear
                    },
                    {
                        'neurons': 1,
                        'activation': None,
                        'type': nn.Linear
                    },
                ]
            }
        }
    ]
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/wells_merged.csv')

    sample_size = 100_000
    train_size = 0.8
    df = wells_merged[['lat', 'lon', 'tvd','country']].copy().dropna(subset=['lat', 'lon', 'tvd']).sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_size), :]
    test_df = df.iloc[int(sample_size*train_size) :, :]

    mtf = ModelTestFramework()
    
    mtf.testModels(
        modelMetas=modelMetas,
        x_train=train_df[['lat', 'lon']],
        y_train=train_df['tvd'],
        x_test=test_df[['lat', 'lon']],
        y_test=test_df['tvd'],
    )