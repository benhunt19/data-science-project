import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge
from src.models.modelClassBase import Model
import warnings

from src.globals import DATA_FOLDER, WELLS_MERGED, WELLS_MERGED_US_CANADA

warnings.simplefilter("ignore", UserWarning)

class NetworkTheoreticRegression(Model):
    """
    Description:
        A regression model using network theory. Constructs a similarity graph based on the input features
        (latitude, longitude) and performs regression with Laplacian regularization.
    Parameters:
        gamma (float): Parameter for the RBF kernel to compute similarity.
        alpha (float): Regularization strength for the Ridge regression.
        k_neighbors (int): Number of neighbors to connect in the graph.
    """
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0, k_neighbors: int = 5):
        super().__init__()
        self.gamma = gamma                          # RBF kernel parameter
        self.alpha = alpha                          # Regularization for Ridge regression
        self.k_neighbors = k_neighbors              # Number of neighbors for the graph
        self.model = Ridge(alpha=self.alpha)        # Base regression model
        self.graph = None                           # To store the similarity graph
        self.laplacian = None                       # Graph Laplacian for regularization

    def _build_graph(self, x: pd.DataFrame) -> None:
        """
        Build a similarity graph using the input features.
        Nodes are data points, edges are based on similarity (RBF kernel).
        """
        # Convert features to numpy array
        X = x.to_numpy()
        n_samples = X.shape[0]

        # Compute similarity matrix using RBF kernel
        similarity_matrix = rbf_kernel(X, X, gamma=self.gamma)

        # Build a k-NN graph: connect each node to its k nearest neighbors
        self.graph = nx.Graph()
        for i in range(n_samples):
            # Get indices of k nearest neighbors (excluding self)
            sim_i = similarity_matrix[i]
            neighbors = np.argsort(-sim_i)[1:self.k_neighbors + 1]  # Top k (excluding self)
            for j in neighbors:
                self.graph.add_edge(i, j, weight=sim_i[j])

        # Compute the graph Laplacian
        self.laplacian = nx.laplacian_matrix(self.graph).toarray()

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train the model by building a graph and fitting a regularized regression.
        """
        self.x_train = x_train
        self.y_train = y_train
        
        self._build_graph(self.x_train)

        

    def test(self, x_test: pd.DataFrame) -> None:
        """
        Predict TVD values for the test set.
        """
        return self.model.predict(x_test)

    def plot(self):
        """
        Plot the similarity graph.
        """
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_size=50, node_color='blue', edge_color='gray', alpha=0.6)
        plt.title("Similarity Graph of Training Data (Nodes: Wells, Edges: Similarity)", pad=20, fontsize=14)
        plt.show()


    @staticmethod
    def __model_name__():
        return 'GraphRegressionModel'
    
    
if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 1000
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    kwargs = {}
    
    ntr = NetworkTheoreticRegression(**kwargs)
    
    ntr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    ntr.plot()