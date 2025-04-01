from src.models.modelClassBase import Model

# Well depeth K nearest neighbour regression
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.globals import DATA_FOLDER
import warnings

warnings.simplefilter("ignore", UserWarning)

class KNNRegressionModel(Model):
    """
    Description:
        Basic wrapper around sklearn KNN Regressor class. Procudes a regression based on nearest neighbours
    Parameters:
        k (int): Number of nearest neighbours to find before running the regression
        weights (str or function): Defauly is 'uniform', can be 'distance' to be inversly related to distance, can even be a single parameter function eg lambda x : x**" 
    """
    def __init__(self, k : int, weights : str = 'uniform'):
        super().__init__()
        self.k = k
        
        self.model = KNeighborsRegressor(n_neighbors=self.k, weights=weights)
        
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame):
        self.model.fit(X=x_train, y=y_train)

    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        return self.model.predict(x_test)
    
    @staticmethod
    def __model_name__():
        return 'KNNRegressionModel'

if __name__ == "__main__":
    
    train_size = 300_000
    test_size = 5_000
    
    wells_merged = pd.read_csv(f'{DATA_FOLDER}/wells_merged.csv')

    train_knn_df = wells_merged[['lat', 'lon', 'tvd','country']].copy().dropna(subset=['lat', 'lon', 'tvd']).sample(train_size).reset_index(drop=True)

    # Number of neighbours (hyperparameter)

    K = [n for n in range(10)]
    error_per_k = []
    sd_per_k = []

    # Create a KNN model
    # accuracy = knn.score(test_knn_df[['lat', 'lon']], test_knn_df['tvd'])

    # Setup the plot
    plt.figure(figsize=(10, 6))
    plt.title('Distribution of Error Percentages in Depth Prediction')
    plt.xlabel('Error Percentage')
    plt.ylabel('Count')


    # For different values of k, test the KNN algorithm
    for k in K:
        
        # Define and train the knn model
        knn = KNNRegressionModel(k=k)
        knn.train(
            x_train=train_knn_df[['lat', 'lon']],
            y_train=train_knn_df['tvd']
        )
        
        accuracy = np.array([0.0 for i in range(test_size)])
        error_pcnt = np.array([0.0 for i in range(test_size)])
        
        # Reset the test dataset    
        test_knn_df = wells_merged[['lat', 'lon', 'tvd','country']].copy().dropna(subset=['lat', 'lon', 'tvd']).sample(test_size).reset_index(drop=True)

        
        for i in range(test_size):
            
            # Get random index
            random_idx = np.random.choice(test_knn_df.index)

            # Get coordinates for this point
            test_point = test_knn_df.loc[random_idx, ['lat', 'lon']].values.reshape(1, -1)
            true_depth = test_knn_df.loc[random_idx, 'tvd']
            country = test_knn_df.loc[random_idx, 'country']

            # Predict depth using KNN
            predicted_depth = knn.test(test_point, true_depth)
            
            print('predicted_depth', predicted_depth)
            print('true_depth', true_depth)
            
            accuracy[i] = (1 - abs(predicted_depth - true_depth) / true_depth) * 100
            error_pcnt[i] = abs(predicted_depth - true_depth) / true_depth * 100


        print(f"Mean percentage accuracy : {accuracy.mean()}")
        print(f"Standard Deviation: {accuracy.std()}")
        error_per_k.append(error_pcnt.mean())
        sd_per_k.append(error_pcnt.std())
        
        # Plotting histogram of error percentage
        error_pcnt_filtered = error_pcnt[error_pcnt < np.percentile(error_pcnt, 90)]
        plt.hist(error_pcnt_filtered, bins=40, edgecolor='black', alpha=0.2)

    plt.show()