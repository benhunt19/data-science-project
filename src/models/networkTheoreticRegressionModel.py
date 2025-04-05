import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pprint import pprint
from sklearn.neighbors import NearestNeighbors
from src.models.modelClassBase import Model
import warnings
import plotly.graph_objects as go


from src.utils import WorldMap
from src.globals import DATA_FOLDER, WELLS_MERGED, WELLS_MERGED_US_CANADA

warnings.simplefilter("ignore", UserWarning)

class NetworkTheoreticRegressionModel(Model):
    """
    Description:
        A regression model using network theory. Constructs a similarity graph based on the input features
        (latitude, longitude) and performs regression with Laplacian regularization.
    Parameters:
        alpha (float): Regularization strength for the Ridge regression.
        k_neighbors (int): Number of neighbors to connect in the graph.
    """
    def __init__(self, gamma: float = 1.0, alpha: float = 1.0, k_neighbors: int = 5, alpha_tvd: float = 0.0005, k_predict_override: int = None):
        super().__init__()
        self.k_neighbors = k_neighbors                  # Number of neighbors for the graph
        self.graph = None                               # To store the similarity graph
        self.distances = None                           # Distances to nearest neighbors
        self.indices = None                             # Indices of nearest neighbors
        self.degree_centrality_vals = None              # Degree centrality values
        self.tvd_centrality_vals = None                 # TVD centrality values
        self.alpha_tvd = alpha_tvd                      # TVD centrality parameter
        self.k_predict_override = k_predict_override    # Number of neighbors for the prediction
        
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
    
    def tvdCentrality(self,):
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
            abs_diff_from_mean = abs(self.y_train.iloc[self.indices[i]].mean() - self.y_train.iloc[i]) # the absolute difference between the mean of the neighbors and the well itself
            tvd_centrality_vals.append( np.exp( -1*abs_diff_from_mean /  (self.y_train.iloc[self.indices[i]].var() * self.alpha_tvd + 10e-8) ) )
                
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

    def test(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """
        Predict TVD values for the test set using weighted k-nearest neighbors based on TVD centrality.
        
        Args:
            x_test: Test features DataFrame containing 'lat' and 'lon'
            y_test: True TVD values for test set
            k_override: Optional override for number of neighbors to use
        """
        test_k_neighbors = self.k_predict_override if self.k_predict_override is not None else self.k_neighbors
        
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
        
        # Calculate weighted predictions with both centrality and distance weighting
        # Convert distances to weights (inverse of distance)
        distance_weights = 1.0 / (distances + 1e-10)  # Adding small epsilon to avoid division by zero
        
        # Combine centrality weights with distance weights
        combined_weights = neighbor_centralities * distance_weights
        
        # Apply combined weights to neighbor TVD values
        weighted_tvd = neighbor_tvd * combined_weights
        y_pred = np.sum(weighted_tvd, axis=1) / np.sum(combined_weights, axis=1)
        
        
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
        scale_factor = 200
        node_sizes = [centrality * scale_factor for centrality in self.tvd_centrality_vals]
        
        # Extract coordinates for plotting
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        
        fig, ax = plt.subplots(figsize=(20, 18))
        
        # Plot world map first (as background)
        wm = WorldMap.worldMapPlot()
        wm.plot(ax=ax, linewidth=1, color='black')
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
        
        # Apply seaborn styling more forcefully
        sns.set_style('darkgrid')
        sns.despine(fig=fig, left=True, bottom=True)
        
        # Add grid with seaborn styling
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Further enhance with seaborn color palette
        plt.rcParams.update(sns.plotting_context("notebook"))
        sns.set_palette("viridis")
        
        # Adjust layout and display
        # change to be kwarg
        plt.tight_layout()
        plt.xlim(-130, -55)
        plt.ylim(25, 65)
        plt.show()
    
    def plotTvdCentrality3D(self):
        """
        Create an interactive 3D visualization of the TVD centrality using Plotly.
        This allows for rotating, zooming, and hovering over wells to see details.
        """
        # Get node positions (longitude and latitude)
        node_positions = {node: (self.graph.nodes[node]['lon'], self.graph.nodes[node]['lat']) 
                         for node in self.graph.nodes()}
        
        # Get centrality values if not already calculated
        if self.tvd_centrality_vals is None:
            self.tvdCentrality()
        
        # Scale centrality values for better visualization
        scale_factor = 500
        node_sizes = [centrality * scale_factor for centrality in self.tvd_centrality_vals]
        
        # Extract coordinates for plotting
        x_coords = [pos[0] for pos in node_positions.values()]
        y_coords = [pos[1] for pos in node_positions.values()]
        
        # Create a DataFrame for easier plotting with Plotly
        df = pd.DataFrame({
            'lon': x_coords,
            'lat': y_coords,
            'tvd': self.y_train.values,
            'centrality': self.tvd_centrality_vals,
            'size': node_sizes
        })
        
        # Create the 3D scatter plot
        fig = go.Figure()
        
        # Add the wells as 3D scatter points
        fig.add_trace(go.Scatter3d(
            x=df['lon'],
            y=df['lat'],
            z=df['tvd'],
            mode='markers',
            marker=dict(
                size=df['size'] / 20,  # Scale down for 3D visualization
                color=df['centrality'],
                colorscale='Viridis',
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=[f"Centrality: {c:.2f}" for c in df['centrality']],
            hoverinfo='text+x+y+z'
        ))
        
        # Add edges as lines
        for edge in self.graph.edges():
            node1_pos = node_positions[edge[0]]
            node2_pos = node_positions[edge[1]]
            
            fig.add_trace(go.Scatter3d(
                x=[node1_pos[0], node2_pos[0]],
                y=[node1_pos[1], node2_pos[1]],
                z=[self.y_train.iloc[edge[0]], self.y_train.iloc[edge[1]]],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.2)', width=1),
                hoverinfo='none'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D TVD Centrality Visualization (k={self.k_neighbors}, sample size={len(self.x_train)})',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='TVD Centrality',
                xaxis=dict(range=[-130, -55]),
                yaxis=dict(range=[25, 65]),
                zaxis=dict(title='TVD Centrality Score')
            ),
            width=1000,
            height=800,
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Add a colorbar
        fig.update_layout(
            coloraxis=dict(
                colorbar=dict(
                    title=dict(text=f'TVD Centrality Score (x{scale_factor})', side='right'),
                    len=0.75
                )
            )
        )
        
        # Flip the z-axis to have values going down from zero
        fig.update_layout(
            scene=dict(
                zaxis=dict(
                    autorange="reversed",  # This flips the z-axis
                    title='Total Vertical Depthh'
                )
            )
        )        
        
        # Show the plot
        fig.show()
        
        return fig
    
    @staticmethod
    def __model_name__():
        return 'GraphRegressionModel'
    
    
if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 1_00
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    kwargs = {
        'alpha_tvd': 0.0005,
        'k_neighbors': 10
    }
    
    ntr = NetworkTheoreticRegressionModel(**kwargs)
    
    ntr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    
    ntr.test(
        x_test=test_df[['lat', 'lon',]],
        y_test=test_df['tvd'],
        k_override=10
    )
    
    # ntr.plotTvdCentrality()
    ntr.plotTvdCentrality3D()
