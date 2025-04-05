from src.models.knnRegressionModel import KNNRegressionModel
from src.models.nnModel import NNModel
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel
from src.models.countryAverageModel import CountryAverageModel
from src.models.hierarchicalRegressionModel import HierarchicalRegressionModel
from src.models.kMeansRegressionModel import KMeansRegressionModel
from src.models.networkTheoreticRegressionModel import NetworkTheoreticRegressionModel
from src.models.modelClassBase import Model
import json as JSON

from src.testing.modelMetaMaker import ModelMetaMaker as MMM

import matplotlib.pyplot as plt
import seaborn as sns

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
        
        self.meta_store = pd.DataFrame([])
        
        # Run testModel over all models in the provided meta
        for i, modelMeta in enumerate(modelMetas):
            print("Testing Model:")
            pprint(modelMeta)
            model_name = modelMeta['model'].__model_name__() + str(i + 1)
            self.results[model_name] = self.testModel(
                modelMeta=modelMeta,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                plot=plot
            )
            # Create a serializable version of the model metadata
            serializable_meta = {
                'model_name': model_name,
                'kwargs': modelMeta['kwargs']
            }
            # Convert the serializable metadata to JSON
            model_meta_str = JSON.dumps(serializable_meta)
            self.meta_store = pd.concat([self.meta_store, pd.DataFrame([model_meta_str])], ignore_index=True)
            
        print(self.results)
        return self
    
    def modelParameterValidation(self, modelMetas : dict, data : pd.DataFrame, resampleCount : int = 10, sampleSize : int = 20_000, trainPcnt : float = 0.8, metric:str = 'MAE', rawParams : dict = None, plot : bool = False) -> None:
        """
        Description:
            Validate the model parameters
        Parameters:
            modelMetas (list[dict]): The model metadata
            data (pd.DataFrame): The training data
            resampleCount (int): The number of times to resample the data
        """
        
        results = []

        # Add as params
        LON = 'lon'
        LAT = 'lat'
        TVD = 'tvd'
        
        # For each model
        for i, modelMeta in enumerate(modelMetas):
            pprint(modelMeta)
            # Resample the data
            sample_results = []
            for i in range(resampleCount):
            
                df = data.sample(sampleSize).reset_index(drop=True)
                
                train_df = df.iloc[:int(sampleSize*trainPcnt), :]
                test_df = df.iloc[int(sampleSize*trainPcnt) :, :]
                
                x_train = train_df[[LON, LAT]]
                y_train = train_df[TVD]
                x_test = test_df[[LON, LAT]]
                y_test = test_df[TVD]
                
                sample_values = self.testModel(
                    modelMeta=modelMeta,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    plot=False
                )
                
                sample_cost = self.metrics()[metric](y_test, sample_values)
                sample_results.append(sample_cost)
            
            results.append(np.mean(sample_results))
            
        paramsName = list(rawParams.keys())[0]
        params = rawParams[paramsName]
        
        RESULTS = 'results'
        results = pd.DataFrame({paramsName: params, RESULTS: results})
        print(results)

        if plot:
            # Create a DataFrame for plotting to ensure 1-dimensional arrays
            plot_df = pd.DataFrame({
                paramsName: params,
                metric: results[RESULTS].values.flatten()  # Flatten to ensure 1D array
            })
            # Use lineplot instead of barplot for a line graph
            sns.lineplot(x=paramsName, y=metric, data=plot_df, marker='o')
            sns.set_theme(style="darkgrid")
            plt.title(f'{modelMetas[0]["model"].__model_name__()} {metric} vs {paramsName}, sample size: {sampleSize}, trainPcnt: {trainPcnt*100}%, resampleCount: {resampleCount}')
            plt.ylabel(metric)
            plt.xlabel(paramsName)  
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
            
        return results
    
    def multiModelParameterValidation(self, model : Model, data : pd.DataFrame, resampleCount : int = 10, sampleSize : int = 20_000, trainPcnt : float = 0.8, metric:str = 'MAE', plot : bool = False, primaryParam : dict = None, secondaryParam : dict = None) -> None:
        """
        Description:
            Validate the model parameters for multiple models
        """
        
        allResults = []
        
        # For each secondary param, for example weighting 
        for secondary in list(secondaryParam.values())[0]:
            
            # Create meta for each secondary param
            modelMetas = MMM.createMeta(
                model=model, 
                kwargs={
                    list(primaryParam.keys())[0]: list(primaryParam.values())[0],
                    list(secondaryParam.keys())[0]: secondary
                }
            )
            
            allResults.append(self.modelParameterValidation(
                modelMetas=modelMetas,
                data=data,
                resampleCount=resampleCount,
                sampleSize=sampleSize,
                trainPcnt=trainPcnt,
                rawParams=primaryParam,
                metric=metric,
                plot=False
            ))
        
        for results in allResults:
            print(results)
        
        
        
        if plot:
            
            paramsName = list(primaryParam.keys())[0]
            
            RESULTS = 'results' # dupe from above
            
            # Create a figure for plotting
            plt.figure(figsize=(10, 6))
            
            # Get the primary parameter name and values
            primaryParamName = list(primaryParam.keys())[0]
            primaryParamValues = list(primaryParam.values())[0]
            
            # Get the secondary parameter name
            secondaryParamName = list(secondaryParam.keys())[0]
            
            # Plot each result set as a separate line
            for i, results in enumerate(allResults):
                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    primaryParamName: primaryParamValues,
                    metric: results[RESULTS].values.flatten()  # Flatten to ensure 1D array
                })
                
                # Get the current secondary parameter value for the legend
                secondaryValue = secondaryParam[secondaryParamName][i]
                label = f"{secondaryParamName}={secondaryValue}"
                
                # Plot the line with a unique color and marker
                sns.lineplot(x=primaryParamName, y=metric, data=plot_df, marker='o', label=label)
            
            # Set plot styling and labels
            sns.set_theme(style="darkgrid")
            plt.title(f'{modelMetas[0]["model"].__model_name__()} {metric} vs {paramsName}, sample size: {sampleSize}, trainPcnt: {trainPcnt*100}%, resampleCount: {resampleCount}')
            plt.ylabel(metric)
            plt.xlabel(primaryParamName)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title=secondaryParamName)
            plt.tight_layout()
            plt.show()
            
        return allResults
    
    
    def metrics(self):
        """
        Description:
            Return a dictionary of evaluation metrics
        """
        return {
            'MSE': mean_squared_error,
            'R2': r2_score,
            'MSLE': mean_squared_log_error,
            'MAPE': lambda y_true, y_pred : np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'MAE': mean_absolute_error,
            'KLD': lambda y_true, y_pred : entropy(y_true, y_pred, base=2)
        }    
        
    
    def evaluateResults(self) -> None:
        """
        Description:
            Perform various evaluation metrics on the results
        """
        
        assert len(self.results) > 0, 'Please run testModels as there are results before running this method'
        
        self.evaluationMetrics = pd.DataFrame()
        
        for column in self.results.columns[1:]:
            metricDict = {}
            for metric_name, metric_func in self.metrics().items():
                metricDict[metric_name] = metric_func(self.results[Y_TEST], self.results[column])
            
            self.evaluationMetrics[column] = pd.Series(metricDict)
        
        print(self.evaluationMetrics)
        return self

if __name__ == "__main__":
    
    knn_meta = MMM.createMeta(model=KNNRegressionModel, kwargs={
        'k': [5, 10, 20, 30],
        'weights': ['distance', None]
    })
    
    dt_meta = MMM.createMeta(model=DecisionTreeRegressionModel, kwargs={
        'maxDepth': [5, 10, 15, 20, 30],
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
    
    # This is busted
    hcr_meta = MMM.createMeta(model=HierarchicalRegressionModel, kwargs={
        'clusters': [10, 20, 30],
        'model_type': ['regression', 'average']
    })
    
    kmr_meta = MMM.createMeta(model=KMeansRegressionModel, kwargs={
        'k': [1, 3, 5, 7, 9, 11, 15, 20, 30, 40, 50],
        'model_type': ['regression', 'average']
    })
    
    ntr_meta = MMM.createMeta(model=NetworkTheoreticRegressionModel, kwargs={
        'k_neighbors': [15, 20, 25],
        'k_predict_override': [10],
        'alpha_tvd': [0.01, 0.02]
    })
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')
    # wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 40_000
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]

    mtf = ModelTestFramework()
    
    # mtf.testModels(
    #     modelMetas=  ntr_meta,
    #     x_train=train_df[['lat', 'lon',]],
    #     y_train=train_df['tvd'],
    #     x_test=test_df[['lat', 'lon', ]],
    #     y_test=test_df['tvd']
    # )
    
    # mtf.evaluateResults()
    
    # mtf.results.to_csv(f'{RESULTS_FOLDER}/results.csv', index=False)
    # mtf.evaluationMetrics.to_csv(f'{RESULTS_METRICS_FOLDER}/evaluationMetrics.csv', index=False)
    # mtf.meta_store.to_csv(f'{RESULTS_METRICS_FOLDER}/metaStore.csv', index=False)
    
    mtf.modelParameterValidation(
        modelMetas=knn_meta,
        data=wells_merged_clean,
        sampleSize=40_000,
        trainPcnt=0.8,
        resampleCount=10,
        metric='MSE'
    )
