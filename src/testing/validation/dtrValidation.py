from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM


wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = 25_000
train_pcnt = 0.80
resample_count = 10

max_depth = [5, 6, 7, 8, 9, 10]
min_samples_split = [2, 3, 4, 5, 10, 15, 20]

dtr_meta = MMM.createMeta(model=DecisionTreeRegressionModel, kwargs={
    'max_depth': max_depth,
    'min_samples_split': min_samples_split
})

mtf = ModelTestFramework()

mtf.multiModelParameterValidation(
    model=DecisionTreeRegressionModel,
    data=wells_merged_clean,
    resampleCount=resample_count,
    sampleSize=sample_size,
    trainPcnt=train_pcnt,
    metric='MSE',
    plot=True,
    primaryParam={'maxDepth': max_depth},
    secondaryParam={'minSamplesSplit': min_samples_split}
)