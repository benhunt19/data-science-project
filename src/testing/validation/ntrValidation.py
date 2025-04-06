from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED, WELLS_MERGED_US_CANADA
from src.models.networkTheoreticRegressionModel import NetworkTheoreticRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM


wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = 25_000
train_pcnt = 0.80
resample_count = 10

alpha_tvd = [0.01, 0.05, 0.1]
k_neighbors = [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]

mtf = ModelTestFramework()

mtf.multiModelParameterValidation(
    model=NetworkTheoreticRegressionModel,
    data=wells_merged_clean,
    resampleCount=resample_count,
    sampleSize=sample_size,
    trainPcnt=train_pcnt,
    metric='MAE',
    plot=True,
    primaryParam={'k_neighbors': k_neighbors},
    secondaryParam={'alpha_tvd': alpha_tvd}
)