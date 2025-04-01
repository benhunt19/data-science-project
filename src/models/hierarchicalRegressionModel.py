import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
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
# Limit Const
HARD_LIMIT = 10_000

class HierarchicalRegressionModel:
    """
    Description:
        Base parent class for use with ML models using hierarchical clustering
    """
    def __init__(self, clusters: int = 3, model_type: str = REGRESSION, linkage_method: str = 'ward'):
        self.models = {}
        self.clusters = clusters
        self.x_train = None
        self.y_train = None
        self.x_cols = None
        self.cluster_centers = None
        self.linearRegModels = {}
        self.type = model_type
        self.linkage_matrix = None
        self.linkage_method = linkage_method  # 'ward', 'single', 'complete', 'average', etc.

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        
        # Check for training length, added as the algorithm scales horribly with time / space
        if len(x_train) > HARD_LIMIT:
            ValueError(f'Exceeded training limit: {HARD_LIMIT}')
        # Copy training data
        self.x_train = x_train.copy()
        self.x_cols = x_train.columns
        self.y_train = y_train.copy()

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(self.x_train, method=self.linkage_method)
        clusters = fcluster(self.linkage_matrix, t=self.clusters, criterion='maxclust')
        self.x_train[CLUSTERS] = clusters - 1  # Adjust to 0-based indexing

        # Compute cluster centers (mean of points in each cluster)
        self.cluster_centers = self.x_train.groupby(CLUSTERS)[self.x_cols].mean()

        # Train linear regression models for each cluster
        if self.type == REGRESSION:
            for cluster_id in range(self.clusters):
                x_train_cluster = self.x_train[self.x_train[CLUSTERS] == cluster_id][self.x_cols]
                y_train_cluster = self.y_train[self.x_train[CLUSTERS] == cluster_id]
                self.linearRegModels[cluster_id] = LinearRegression().fit(x_train_cluster, y_train_cluster)

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> np.ndarray:
        # Copy test data
        x_test_trans = x_test.copy().reset_index(drop=True)

        # Predict clusters for test data by finding nearest cluster center
        predicted_clusters = self._assign_clusters(x_test_trans)

        # Initialize prediction column
        x_test_trans[PREDICTION] = 0.0

        # Perform predictions based on the model type
        if self.type == REGRESSION:
            for cluster_id in range(self.clusters):
                mask = predicted_clusters == cluster_id
                cluster_x_test = x_test_trans.loc[mask, self.x_cols]
                
                # Adjust for edge case where there are no items in the cluster?
                if len(cluster_x_test) > 0:
                    predictions = self.linearRegModels[cluster_id].predict(x_test_trans.loc[mask, self.x_cols])
                    x_test_trans.loc[mask, PREDICTION] = np.maximum(predictions, 0.001)
                else:
                    pass

        elif self.type == AVERAGE:
            for cluster_id in range(self.clusters):
                # Get the mean of training data for this cluster
                train_mask = self.x_train[CLUSTERS] == cluster_id
                cluster_mean = self.y_train[train_mask].mean()

                # Apply the mean to test data points in this cluster
                test_mask = predicted_clusters == cluster_id
                predictions = np.zeros(len(x_test_trans.loc[test_mask, PREDICTION])) + cluster_mean
                
                # Adjust for edge case where there are no items in the cluster?
                if len(predictions) > 0:
                    x_test_trans.loc[test_mask, PREDICTION] = np.maximum(predictions, 0)
                else:
                    pass
                
        else:
            raise KeyError("Incorrect Type")

        # print(x_test_trans[PREDICTION].to_numpy())
        # print(y_test)
        return x_test_trans[PREDICTION].to_numpy()

    def _assign_clusters(self, x_test: pd.DataFrame) -> pd.Series:
        # Assign test points to the nearest cluster center
        distances = np.zeros((len(x_test), self.clusters))
        for cluster_id in range(self.clusters):
            center = self.cluster_centers.loc[cluster_id].values
            distances[:, cluster_id] = np.sum((x_test[self.x_cols] - center) ** 2, axis=1)
        return pd.Series(np.argmin(distances, axis=1))

    @staticmethod
    def __model_name__():
        return 'HierarchicalRegression'

if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd', 'country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    canada = wells_merged_clean[wells_merged_clean['country'] == 'Canada'].copy()
    del wells_merged, wells_merged_clean
    
    sample_size = 20_000
    train_pcnt = 0.8
    
    df = canada.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size * train_pcnt), :]
    test_df = df.iloc[int(sample_size * train_pcnt):, :]
    
    clusters = 10
    hrm = HierarchicalRegressionModel(clusters=clusters, linkage_method='ward')
    hrm.train(
        x_train=train_df[['lat', 'lon']],
        y_train=train_df['tvd']
    )
    
    hrm.test(
        x_test=test_df[['lat', 'lon']],
        y_test=test_df['tvd']
    )