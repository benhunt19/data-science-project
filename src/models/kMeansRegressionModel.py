import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from src.globals import DATA_FOLDER, WELLS_MERGED
from pprint import pprint
import math

# Type Consts
REGRESSION = 'regression'
AVERAGE = 'average'

# Column consts
PREDICTION = 'prediction'
CLUSTERS = 'clusters'

class KMeansRegressionModel:
    """
    Description:
        Base parent class for use with ML models
    """
    def __init__(self, n_clusters : int = 10, type : str = REGRESSION):
        self.model = None
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.type = type
        self.x_cols = []

    def train(self, x_train, y_train):
        
        self.x_train = x_train.copy()
        self.x_cols = x_train.columns
        del x_train
        
        self.y_train = y_train.copy()
        del y_train
        
        n_clusters = 20
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        
        clusters = self.model.fit_predict(self.x_train)
        self.x_train[CLUSTERS] = clusters
        
        self.cluster_centers = pd.DataFrame(self.model.cluster_centers_)
        
        # Perform a linear regression on all the of clusters
        self.linearRegModels = {}
        
        if self.type == REGRESSION:
            for row in self.cluster_centers.itertuples():
                id = row[0] # Index
                x_train_cluster = self.x_train[self.x_train[CLUSTERS] == id][self.x_cols]
                y_train_cluster = self.y_train[self.x_train[CLUSTERS] == id]
                self.linearRegModels[id] = LinearRegression().fit(x_train_cluster, y_train_cluster)
    
    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame) -> None:
        
        # PREDICTED_CLUSTERS = 'predicted_clusters'
        x_test_trans = x_test.copy().reset_index(drop=True)
        del x_test
        predictedClusters = pd.Series(self.model.predict(x_test_trans))
        
        x_test_trans[PREDICTION] = 0.0  # Initialize prediction column
        
        # Type 1, regression
        if self.type == REGRESSION:
            
            for cluster in self.cluster_centers.itertuples():
                id = cluster[0]
                mask = predictedClusters == id
                predictions = self.linearRegModels[id].predict(x_test_trans.loc[mask, self.x_cols])
                x_test_trans.loc[mask, PREDICTION] = np.maximum(predictions, 0.001)
                
        # Type 2, average    
        elif self.type == AVERAGE:
            for cluster in self.cluster_centers.itertuples():
                id = cluster[0]
                # Get the mean of training data for this cluster
                train_mask = self.x_train[CLUSTERS] == id
                cluster_mean = self.y_train[train_mask].mean()
                
                # Apply the mean to test data points in this cluster
                test_mask = predictedClusters == id
                predictions = np.zeros(len(x_test_trans.loc[test_mask, PREDICTION])) + cluster_mean
                x_test_trans.loc[test_mask, PREDICTION] = np.maximum(predictions, 0)
        else:
            KeyError("Incorrect Type")
            
        return x_test_trans[PREDICTION].to_numpy()
            

    @staticmethod
    def __model_name__():
        return 'KMeansRegression'
    

if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

    
    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 100_000
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    clusters = 10
    kmr = KMeansRegressionModel(n_clusters=clusters)
    kmr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    kmr.test(
        x_test=test_df[['lat', 'lon',]],
        y_test=test_df['tvd']
    )