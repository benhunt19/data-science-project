import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from src.globals import DATA_FOLDER, WELLS_MERGED, WELLS_MERGED_US_CANADA
from pprint import pprint
import matplotlib.pyplot as plt
import math
import seaborn as sns

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
    def __init__(self, k: int = 3, model_type: str = REGRESSION):
        self.models = {}
        self.k = k
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
        self.model = KMeans(n_clusters=self.k)
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
                cluster_x_test = x_test_trans.loc[mask, self.x_cols]
                
                # Check that exists some testing data in the training cluster
                if len(cluster_x_test) > 0:
                    predictions = self.linearRegModels[cluster_id].predict(cluster_x_test)
                    x_test_trans.loc[mask, PREDICTION] = np.maximum(predictions, 0.001)
                else:
                    pass

        elif self.type == AVERAGE:
            for cluster in self.cluster_centers.itertuples():
                cluster_id = cluster[0]
                # Get the mean of training data for this cluster
                train_mask = self.x_train[CLUSTERS] == cluster_id
                cluster_mean = self.y_train[train_mask].mean()

                # Apply the mean to test data points in this cluster
                test_mask = predicted_clusters == cluster_id
                cluster_x_test = x_test_trans.loc[test_mask, PREDICTION]
                if len(cluster_x_test) > 0:
                    predictions = np.zeros(len(cluster_x_test)) + cluster_mean
                    x_test_trans.loc[test_mask, PREDICTION] = np.maximum(predictions, 0)

        else:
            raise KeyError("Incorrect Type")

        return x_test_trans[PREDICTION].to_numpy()
    
    def plotCluster(self, cluster_id):
        """
        Description:
            3D plot of the well depths for a cluster
        Parameter:
            The cluster number of the cluster to plot. cluster_id < k.
        """
        # Get data for the specified cluster
        cluster_mask = self.x_train[CLUSTERS] == cluster_id
        cluster_data = self.x_train[cluster_mask]
        cluster_tvd = self.y_train[cluster_mask]  # Removed negation
        # Get predictions from the linear regression model for this cluster
        X_cluster = cluster_data[self.x_cols]
        y_pred = self.linearRegModels[cluster_id].predict(X_cluster)

        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create a scatter plot of the predicted values
        # Create a grid of points
        lon_min, lon_max = cluster_data['lon'].min(), cluster_data['lon'].max()
        lat_min, lat_max = cluster_data['lat'].min(), cluster_data['lat'].max()
        lon_grid, lat_grid = np.meshgrid(
            np.linspace(lon_min, lon_max, 50),
            np.linspace(lat_min, lat_max, 50)
        )
        
        # Create prediction points for the surface
        grid_points = pd.DataFrame({
            col: lon_grid.ravel() if col == 'lon' else lat_grid.ravel()
            for col in self.x_cols
        })
        
        # Get predictions for the grid
        z_pred = self.linearRegModels[cluster_id].predict(grid_points)
        z_grid = z_pred.reshape(lon_grid.shape)  # Removed negation
        
        # Plot the surface
        surface = ax.plot_surface(
            lon_grid, lat_grid, z_grid,
            alpha=0.5,
            cmap='viridis',
            label='Predicted Surface'
        )
        ax.legend()
        
        # Plot scatter points
        scatter = ax.scatter(
            cluster_data['lon'], 
            cluster_data['lat'], 
            cluster_tvd,
            c=cluster_tvd,  # Color by depth
            cmap='viridis'
        )

        # Labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('True Vertical Depth (m)')
        ax.set_title(f'Well Locations and Depth - Cluster {cluster_id}')
        
        # Add colorbar
        plt.colorbar(scatter, label='Depth (m)')
        # Invert z-axis
        ax.invert_zaxis()
        plt.show()

    
    def plotResiduals(self, cluster_ids):
        """
        Creates histograms of residuals (predicted - actual) for specified clusters in subplots
        Args:
            cluster_id: int or list of ints representing cluster(s) to plot
        """
        # Convert single cluster_id to list for consistent handling
        if isinstance(cluster_id, int):
            cluster_ids = [cluster_id]
        else:
            cluster_ids = cluster_id
            
        # Calculate number of rows and columns for subplots
        n_plots = len(cluster_ids)
        n_cols = min(3, n_plots)  # Maximum 3 columns
        n_rows = math.ceil(n_plots / n_cols)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        sns.set_style("darkgrid")
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each cluster's residuals
        for idx, cluster in enumerate(cluster_ids):
            # Get data for the specified cluster and set dark style
            cluster_mask = self.x_train[CLUSTERS] == cluster
            cluster_data = self.x_train[cluster_mask]
            cluster_tvd = self.y_train[cluster_mask]
            
            # Get predictions and calculate residuals
            predictions = self.linearRegModels[cluster].predict(cluster_data[self.x_cols])
            residuals = (predictions - cluster_tvd)
            
            # Create histogram on subplot
            sns.histplot(data=residuals, bins=50, kde=True, ax=axes[idx])
            axes[idx].set_xlabel('Residual Value (Predicted - Actual)')
            axes[idx].set_ylabel('Count')
            axes[idx].set_title(f'Residuals Distribution - Cluster {cluster}')
            axes[idx].legend(labels=['Distribution', 'KDE'])
            
        # Remove empty subplots if any
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        sns.set_style("dark")
        sns.set_palette("deep")
        plt.show()
        # Get data for the specified cluster
        cluster_mask = self.x_train[CLUSTERS] == cluster_id
        cluster_data = self.x_train[cluster_mask]
        cluster_tvd = self.y_train[cluster_mask]
        
        # Get predictions
        predictions = self.linearRegModels[cluster_id].predict(cluster_data[self.x_cols])
        
        # Calculate residuals
        residuals = (predictions - cluster_tvd)
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        sns.set_style("darkgrid")
        plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Histogram')
        sns.kdeplot(data=residuals, color='red', label='KDE')
        plt.legend()
        # plt.hist(residuals, bins=50, density=True, alpha=0.7)
        plt.xlabel('Residual Value (Predicted - Actual)')
        plt.ylabel('Density')
        plt.title(f'Distribution of Residuals for Cluster {cluster_id}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def __model_name__():
        return 'KMeansRegression'
    

if __name__ == '__main__':
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED_US_CANADA}.csv')

    
    # Potentially add this to a method that handles it internally sklearn.model_selection import train_test_split
    wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
    wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])
    del wells_merged
    
    sample_size = 100_000
    train_pcnt = 0.8
    
    df = wells_merged_clean.sample(sample_size, random_state=42).reset_index(drop=True)
    
    train_df = df.iloc[:int(sample_size*train_pcnt), :]
    test_df = df.iloc[int(sample_size*train_pcnt) :, :]
    
    k = 40
    kmr = KMeansRegressionModel(k=k)
    kmr.train(
        x_train=train_df[['lat', 'lon',]],
        y_train=train_df['tvd']
    )
    
    kmr.test(
        x_test=test_df[['lat', 'lon',]],
        y_test=test_df['tvd']
    )
    
    # cluster_id = np.random.randint(0, kmr.k)
    # cluster_id = 2
    # cluster_id = 3
    cluster_id = 2,3,4
    
    # kmr.plotCluster(cluster_id=cluster_id)
    kmr.plotResiduals(cluster_ids=cluster_id)