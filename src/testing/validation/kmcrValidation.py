from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED
from src.models.kMeansRegressionModel import KMeansRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM

wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = 40_000
train_pcnt = 0.8

k = [2, 3, 5, 7, 9, 11, 15, 20, 30, 40, 50, 75, 100, 200]
model_type = ['regression', 'average']
resample_count = 10

mtf = ModelTestFramework()

mtf.multiModelParameterValidation(
    model=KMeansRegressionModel,
    data=wells_merged_clean,
    resampleCount=resample_count,
    sampleSize=sample_size,
    trainPcnt=train_pcnt,
    metric='MAE',
    plot=True,
    primaryParam={'k': k},
    secondaryParam={'model_type': model_type}
)
