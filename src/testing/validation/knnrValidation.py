from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED
from src.models.knnRegressionModel import KNNRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM


wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = 25_000
train_pcnt = 0.80
resample_count = 10

k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
weights = ['distance', 'uniform']

mtf = ModelTestFramework()

mtf.multiModelParameterValidation(
    model=KNNRegressionModel,
    data=wells_merged_clean,
    resampleCount=resample_count,
    sampleSize=sample_size,
    trainPcnt=train_pcnt,
    metric='MAE',
    plot=True,
    primaryParam={'k': k},
    secondaryParam={'weights': weights}
)
