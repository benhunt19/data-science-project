from src.models.knnRegressionModel import KNNRegressionModel
from src.models.nnModel import NNModel
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel
from src.models.countryAverageModel import CountryAverageModel
from src.models.hierarchicalRegressionModel import HierarchicalRegressionModel
from src.models.kMeansRegressionModel import KMeansRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM

from src.globals import (
    DATA_FOLDER,
    WELLS_MERGED,
    WELLS_MERGED_US_CANADA,
    RESULTS_FOLDER,
    RESULTS_METRICS_FOLDER
)
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    mean_absolute_error,
)
from scipy.stats import entropy
import torch.nn as nn

# Const for test column name
Y_TEST = 'y_test'

class ModelTestFramework:
    def __init__(self):
        self.results = pd.DataFrame([])                 # Dataframe to store prediction results from the model tests
        self.evaluationMetrics = None                   # Datafame to store metrics
    
    def testModel(self, modelMeta : list[dict], x_train: pd.DataFrame, y_train: pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame, plot : bool = False) -> None:
        """
        Description:
            Train model on training data then test the model on the test set
        Parameters:
            modelMeta (list[dict]): The metadata to create a model to test. eg.
                [{
                    'model': HierarchicalRegressionModel,
                    'kwargs': {
                        'clusters': 10
                    }
                }]
            x_train (pd.DataFrame): Training data dataframe
            y_train (pd.DataFrame): Training labels dataframe (or series)
            x_test (pd.DataFrame): Testing data dataframe
            y_test (pd.DataFrame): Testing labels dataframe (or series)
            plot (bool): Run each model's plotting function when testing
            
        """
        # Create a new instance of the model
        m = modelMeta['model'](**modelMeta['kwargs'])
        
        # Train the model on the test data
        m.train(x_train=x_train, y_train=y_train)
        
        # This will require an override on the class to the base class
        if plot:
            m.plot()
        
        # Test the trained model on the testing set
        return m.test(x_test=x_test, y_test=y_test)
        

    def testModels(self, modelMetas : list[dict], x_train: pd.DataFrame, y_train: pd.DataFrame, x_test : pd.DataFrame, y_test : pd.DataFrame, plot : bool = False) -> None:
        """
        Description:
            Run testModel over an array of models with various parameters and kwargs
        Parameters:
            modalMetas (list[dict]): Array of meta data for each model to be tested
        """
        
        # Initiate the actual test data in the results dataframe
        self.results = pd.DataFrame({
            Y_TEST: y_test
        })
        
        # Run testModel over all models in the provided meta
        for i, modelMeta in enumerate(modelMetas):
            print("Testing Model:")
            pprint(modelMeta)
            self.results[modelMeta['model'].__model_name__() + str(i + 1)] = self.testModel(
                modelMeta=modelMeta,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                plot=plot
            )
        print(self.results)
        
        return self
    
    def evaluateResults(self) -> None:
        """
        Description:
            Perform various evaluation metrics on the results
        """
        
        assert len(self.results) > 0, 'Please run testModels as there are results before running this method'
        
        metrics = {
            'MSE': mean_squared_error,
            'R2': r2_score,
            'MSLE': mean_squared_log_error,
            'MAPE': lambda y_true, y_pred : np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'MAE': mean_absolute_error,
            'KLD': lambda y_true, y_pred : entropy(y_true, y_pred, base=2)
        }
        
        self.evaluationMetrics = pd.DataFrame()
        
        for column in self.results.columns[1:]:
            metricDict = {}
            for metric_name, metric_func in metrics.items():
                metricDict[metric_name] = metric_func(self.results[Y_TEST], self.results[column])
            
            self.evaluationMetrics[column] = pd.Series(metricDict)
        
        print(self.evaluationMetrics)
        return self

if __name__ == "__main__":
    
    knn_meta = MMM.createMeta(model=KNNRegressionModel, kwargs={
        'k': [1, 5, 10, 20, 30, 40, 50],
        'weights': ['distance', None]
    })
    
    dt_meta = MMM.createMeta(model=DecisionTreeRegressionModel, kwargs={
        'maxDepth': [10, 20, 30, 40, 50],
        'minSamplesSplit': [10, 20, 30, 40, 50],
    })
    
    ca_meta = [
        {
            'model': CountryAverageModel,
            'kwargs': {}
        }
    ]
    
    nn_meta = [{
        'model': NNModel,
        'kwargs': {
            'n_epochs': 10,
            'learning_rate': 0.01,
            'network_meta': [
                {'neurons': 2},
                {'neurons': 30, 'activation': nn.ReLU, 'type': nn.Linear},
                {'neurons': 50, 'activation': nn.ReLU, 'type': nn.Linear},
                {'neurons': 50, 'activation': nn.ReLU, 'type': nn.Linear},
                {'neurons': 1, 'activation': None, 'type': nn.Linear},
            ]
        }
    }]
    
    hcr_meta = MMM.createMeta(model=HierarchicalRegressionModel, kwargs={
        'clusters': [10, 20, 30],
        'model_type': ['regression', 'average']
    })
    
    kmr_meta = MMM.createMeta(model=KMeansRegressionModel, kwargs={
        'k': [1, 3, 5, 7, 9, 11, 15, 20, 30, 40, 50],
        'model_type': ['regression', 'average']
    })
    
    # wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 100_000
    # sample_size = len(wells_merged_clean) - 1 
    train_pcnt = 0.9
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]

    mtf = ModelTestFramework()
    
    mtf.testModels(
        modelMetas= kmr_meta,
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd'],
        x_test=test_df[['lat', 'lon', ]],
        y_test=test_df['tvd']
    )
    
    mtf.evaluateResults()