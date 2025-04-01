import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from src.models.modelClassBase import Model

# Type consts
REGRESSION = 'regression'
AVERAGE = 'average'

# Column consts
CLUSTERS = 'clusters'
PREDICTION = 'prediction'

class HeirachicalRegressionModel(Model):
    """
    Description:
        Heirachical Clustering Linear Regression Model
    """
    def __init__(self, clusters: int = 3, method: str = 'ward', criterion: str = 'maxclust', model_type: str = REGRESSION):
        self.models = {}
        self.clusters = clusters
        self.method = method
        self.criterion = criterion
        self.x_train = None
        self.y_train = None
        self.x_cols = None
        self.cluster_centers = None
        self.linearRegModels = {}
        self.type = model_type
        self.model = None

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        # Copy training data
        self.x_train = x_train.copy()
        self.x_cols = x_train.columns
        self.y_train = y_train.copy()

        # Perform KMeans clustering
        n_clusters = self.clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.model.fit_predict(self.x_train)
        self.x_train[CLUSTERS] = clusters

        # Store cluster centers
        self.cluster_centers = pd.DataFrame(self.model.cluster_centers_)

        # Train linear regression models for each cluster
        if self.type == REGRESSION:
            for row in self.cluster_centers.itertuples():
                cluster_id = row[0]  # Index
                x_train_cluster = self.x_train[self.x_train[CLUSTERS] == cluster_id][self.x_cols]
                y_train_cluster = self.y_train[self.x_train[CLUSTERS] == cluster_id]
                self.linearRegModels[cluster_id] = LinearRegression().fit(x_train_cluster, y_train_cluster)

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> np.ndarray:
        # Copy test data
        x_test_trans = x_test.copy().reset_index(drop=True)
        predicted_clusters = pd.Series(self.model.predict(x_test_trans))

        # Initialize prediction column
        x_test_trans[PREDICTION] = 0.0

        # Perform predictions based on the model type
        if self.type == REGRESSION:
            for cluster in self.cluster_centers.itertuples():
                cluster_id = cluster[0]
                mask = predicted_clusters == cluster_id
                predictions = self.linearRegModels[cluster_id].predict(x_test_trans.loc[mask, self.x_cols])
                x_test_trans.loc[mask, PREDICTION] = np.maximum(predictions, 0.001)

        elif self.type == AVERAGE:
            for cluster in self.cluster_centers.itertuples():
                cluster_id = cluster[0]
                # Get the mean of training data for this cluster
                train_mask = self.x_train[CLUSTERS] == cluster_id
                cluster_mean = self.y_train[train_mask].mean()

                # Apply the mean to test data points in this cluster
                test_mask = predicted_clusters == cluster_id
                predictions = np.zeros(len(x_test_trans.loc[test_mask, PREDICTION])) + cluster_mean
                x_test_trans.loc[test_mask, PREDICTION] = np.maximum(predictions, 0)

        else:
            raise KeyError("Incorrect Type")

        return x_test_trans[PREDICTION].to_numpy()

    @staticmethod
    def __model_name__():
        return 'HeirachicalRegression'