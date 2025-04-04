import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pprint import pprint
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from src.models.modelClassBase import Model
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


from src.utils import WorldMap
from src.globals import DATA_FOLDER, WELLS_MERGED, WELLS_MERGED_US_CANADA

warnings.simplefilter("ignore", UserWarning)

class NetworkTheoreticRegressionModel(Model):
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
        self.model = None                           # Base regression model
        self.graph = None                           # To store the similarity graph
        self.laplacian = None                       # Graph Laplacian for regularization
        self.distances = None                       # Distances to nearest neighbors
        self.indices = None                         # Indices of nearest neighbors
        self.degree_centrality_vals = None          # Degree centrality values
        self.tvd_centrality_vals = None             # TVD centrality values

    def _build_graph(self, x_train: pd.DataFrame, y_train: pd.DataFrame, hbors: int = 2) -> None:
        """
        Build a similarity graph using the input features.
        Nodes are data points, edges are based on similarity (RBF kernel).
        """
        """
        Build a graph where each node is connected to its nearest neighbor based on longitude and latitude.
        """
        
        self.graph = nx.Graph()
        
        # Add nodes with attributes (latitude, longitude)
        for idx, row in x_train.iterrows():
            self.graph.add_node(idx, lat=row['lat'], lon=row['lon'])
        
        # Extract coordinates (latitude, longitude)
        coords = x_train[['lon', 'lat']].to_numpy()
        
        # Use k-NN to find the nearest neighbor for each node
        # k=2 because the first neighbor is the node itself, the second is the nearest other node
        nn = NearestNeighbors(n_neighbors=self.k_neighbors, algorithm='auto').fit(coords)
        self.distances, self.indices = nn.kneighbors(coords)
        
        # Add edges: connect each node to its nearest neighbor
        for i in range(len(x_train)):
            # Connect to all k nearest neighbors (skipping the first which is the node itself)
            for k in range(1, self.k_neighbors):
                neighbor = self.indices[i, k]
                self.graph.add_edge(i, neighbor, weight=self.distances[i, k])
        
    
    def degreeCentrality(self):
        """
        Calculate the degree centrality of the graph.
        """
        self.degree_centrality_vals = nx.degree_centrality(self.graph)
        return self.degree_centrality_vals
    
    def tvdCentrality(self):
        """
        Description:
            Calculate the centrality of the graph. We will use a custom function to calculate the centrality of the graph.
            The function will be based on the similarity between the TVD of a well and its neighbors.
            The idea is that if a well has a similar TVD to its neighbors, it is more central, and hence can be used to predict the TVD of other wells.
        """
        
        tvd_centrality_vals = []
        
        # Calculate the TVD centrality for each well
        # The centrality is the inverse of the mean absolute 
        # difference between the TVD of a well and its neighbors
        for i in range(len(self.x_train)):
            abs_diff = abs(self.y_train.iloc[self.indices[i]] - self.y_train.iloc[i])
            tvd_centrality_vals.append(1 / np.mean(abs_diff[1:]))
                
        self.tvd_centrality_vals = tvd_centrality_vals
        
        return self.tvd_centrality_vals
        

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Train the model by building a graph and fitting a regularized regression.
        """
        self.x_train = x_train
        self.y_train = y_train
        
        # Build the graph and calculate TVD centrality in one pass
        self._build_graph(self.x_train, self.y_train)
        self.tvdCentrality()

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame, k_override: int = None) -> None:
        """
        Predict TVD values for the test set using weighted k-nearest neighbors based on TVD centrality.
        
        Args:
            x_test: Test features DataFrame containing 'lat' and 'lon'
            y_test: True TVD values for test set
            k_override: Optional override for number of neighbors to use
        """
        test_k_neighbors = k_override if k_override is not None else self.k_neighbors
        
        # Extract coordinates once
        test_coords = x_test[['lon', 'lat']].values
        train_coords = self.x_train[['lon', 'lat']].values
        
        # Find nearest neighbors for all test points at once
        nn = NearestNeighbors(n_neighbors=test_k_neighbors, algorithm='auto').fit(train_coords)
        distances, indices = nn.kneighbors(test_coords)
        
        # Vectorized prediction calculation
        # Get TVD values and centralities for all neighbors at once
        neighbor_tvd = self.y_train.iloc[indices.ravel()].values.reshape(-1, test_k_neighbors)
        neighbor_centralities = np.take(self.tvd_centrality_vals, indices)
        
        # Calculate weighted predictions
        weighted_tvd = neighbor_tvd * neighbor_centralities
        y_pred = np.sum(weighted_tvd, axis=1) / np.sum(neighbor_centralities, axis=1)
        
        # Store predictions and calculate metrics
        # self.y_pred = y_pred
        # self.mse = mean_squared_error(y_test, y_pred)
        # self.r2 = r2_score(y_test, y_pred)
        
        return y_pred

    def plot(self):
        """
        Plot the similarity graph.
        """
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, node_size=50, node_color='blue', edge_color='gray', alpha=0.6)
        plt.title("Similarity Graph of Training Data (Nodes: Wells, Edges: Similarity)", pad=20, fontsize=14)
        plt.show()
        
    def plotNodes(self):
        """
        Plot the wells on a 2D map (latitude vs. longitude) with edges to their nearest neighbors.
        """
        if self.graph is None:
            raise ValueError("Must build the graph before plotting.")
        
        plt.figure(figsize=(15, 10))
        
        # Extract node positions
        lats = [self.graph.nodes[node]['lat'] for node in self.graph.nodes]
        lons = [self.graph.nodes[node]['lon'] for node in self.graph.nodes]
        
        # Plot nodes (wells)
        plt.scatter(lons, lats, c='blue', s=2, alpha=0.7, label='Wells')
        
        # Plot edges (connections to nearest neighbors)
        for edge in self.graph.edges:
            node1, node2 = edge
            lon1, lat1 = self.graph.nodes[node1]['lon'], self.graph.nodes[node1]['lat']
            lon2, lat2 = self.graph.nodes[node2]['lon'], self.graph.nodes[node2]['lat']
            plt.plot([lon1, lon2], [lat1, lat2], 'gray', alpha=0.5)
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Wells in North America: Connections to {self.k_neighbors} Nearest Neighbors')
        plt.grid(True)
        plt.legend()
        
        # if lat_lim is not None:
            # plt.ylim(lat_lim)
        # if long_lim is not None:
            # plt.xlim(long_lim)
        
        plt.show()
    
    def plotDegreeCentrality(self):
        """
        Plot the graph with nodes sized according to their degree centrality.
        """
        plt.figure(figsize=(10, 8))
        
        # Get node positions (longitude and latitude)
        node_positions = {node: (self.graph.nodes[node]['lon'], self.graph.nodes[node]['lat']) 
                         for node in self.graph.nodes()}
        
        # Get centrality values
        centrality_values = self.degreeCentrality()
        
        # Scale centrality values for better visualization (multiply by 500 to make differences more visible)
        node_sizes = [centrality_values[node] * 500 for node in self.graph.nodes()]
        
        # Extract coordinates for plotting
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        
        # Plot nodes with size based on centrality
        scatter = plt.scatter(
            x_coords,
            y_coords,
            s=node_sizes,
            c=node_sizes, 
            cmap='viridis',
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add colorbar to show centrality scale
        cbar = plt.colorbar(scatter)
        cbar.set_label('Centrality Score')
        
        # Add edges from the graph
        for edge in self.graph.edges():
            node1_pos = node_positions[edge[0]]
            node2_pos = node_positions[edge[1]]
            plt.plot(
                [node1_pos[0],
                 node2_pos[0]],
                [node1_pos[1],
                 node2_pos[1]], 
                'k-',
                alpha=0.2,
                linewidth=0.5
            )
        
        plt.title('Well Locations with Centrality Visualization')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    def plotTvdCentrality(self):
        """
        Plot the graph with nodes sized according to their TVD centrality.
        """
        
        # Get node positions (longitude and latitude)
        node_positions = {node: (self.graph.nodes[node]['lon'], self.graph.nodes[node]['lat']) 
                         for node in self.graph.nodes()}
        
        # Get centrality values if not already calculated
        if self.tvd_centrality_vals is None:
            self.tvdCentrality()
        
        # Scale centrality values for better visualization
        scale_factor = 10_000
        node_sizes = [centrality * scale_factor for centrality in self.tvd_centrality_vals]
        
        # Extract coordinates for plotting
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        
        fig, ax = plt.subplots(figsize=(20, 18))
        
        # Plot world map first (as background)
        wm = WorldMap.worldMapPlot()
        wm.plot(ax=ax, linewidth=1, color='black')
        sns.set_style('darkgrid')
        # Plot edges from the graph
        for edge in self.graph.edges():
            node1_pos = node_positions[edge[0]]
            node2_pos = node_positions[edge[1]]
            ax.plot(
                [node1_pos[0], node2_pos[0]],
                [node1_pos[1], node2_pos[1]], 
                'k-',
                alpha=0.2,
                linewidth=0.5
            )
        
        # Plot nodes with size based on centrality
        scatter = ax.scatter(
            x_coords,
            y_coords,
            s=node_sizes,
            c=node_sizes, 
            cmap='viridis',
            alpha=0.7,
            edgecolors='black'
        )
        
        # Add colorbar to show centrality scale
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'TVD Centrality Score (x{scale_factor})')
        
        # Set title and labels
        ax.set_title(f'Well Locations with TVD Centrality Visualization (k={self.k_neighbors}, sample size={len(self.x_train)})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.xlim(-155, -50)
        plt.ylim(25, 75)
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
    
    sample_size = 500
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    kwargs = {
        'k_neighbors': 10
    }
    
    ntr = NetworkTheoreticRegressionModel(**kwargs)
    
    ntr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    # ntr.tvdCentrality()
    # ntr.plotTvdCentrality()
    
    # ntr.plotNodes()
    
    ntr.test(
        x_test=test_df[['lat', 'lon',]],
        y_test=test_df['tvd'],
        k_override=10
    )
    
    ntr.plotTvdCentrality()
