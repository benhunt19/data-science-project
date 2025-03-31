import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.linear_model import LinearRegression

from src.models.modelClassBase import Model

class HeirachicalRegressionModel(Model):
    pass


# Sample data (features + target)
np.random.seed(42)
X = np.random.rand(100, 2)  # 2D features
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.2  # Target variable

# Step 1: Perform hierarchical clustering
Z = linkage(X, method='ward')  # Ward's method minimizes variance
clusters = fcluster(Z, t=3, criterion='maxclust')  # Create 3 clusters

# Step 2: Fit separate regression models per cluster
models = {}
for cluster in np.unique(clusters):
    idx = clusters == cluster
    X_cluster, y_cluster = X[idx], y[idx]

    model = LinearRegression().fit(X_cluster, y_cluster)
    models[cluster] = model  # Store model for each cluster

# Predict for a new data point
new_X = np.array([[0.5, 0.2]])  # Example new data
new_cluster = fcluster(linkage(np.vstack([X, new_X]), method='ward'), t=3, criterion='maxclust')[-1]
predicted_y = models[new_cluster].predict(new_X)

print(f"Predicted y: {predicted_y[0]}")
