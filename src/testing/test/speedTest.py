from src.testing.modelTestFramework import ModelTestFramework

from src.globals import DATA_FOLDER, WELLS_MERGED, TEST_DATA_FOLDER, WELLS_MERGED_TEST, TEST_RESULTS_FOLDER
from src.models.knnRegressionModel import KNNRegressionModel
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel
from src.models.countryAverageModel import CountryAverageModel
from src.models.hierarchicalRegressionModel import HierarchicalRegressionModel
from src.models.kMeansRegressionModel import KMeansRegressionModel
from src.models.networkTheoreticRegressionModel import NetworkTheoreticRegressionModel
from src.models.modelClassBase import Model

from src.testing.modelMetaMaker import ModelMetaMaker as MMM
from datetime import datetime
import pandas as pd

wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = len(wells_merged_clean)
wells_merged_clean_sample = wells_merged_clean.sample(sample_size)

# Baseline
ca_meta = MMM.createMeta(model=CountryAverageModel, kwargs={})

# Models
knnr_meta = MMM.createMeta(model=KNNRegressionModel, kwargs={'k': 7, 'weights': 'distance'})
kmcr_meta = MMM.createMeta(model=KMeansRegressionModel, kwargs={'k': 100, 'model_type': 'regression'})
dtr_meta = MMM.createMeta(model=DecisionTreeRegressionModel, kwargs={'maxDepth': 15,'minSamplesSplit': 15})
ntr_meta = MMM.createMeta(model=NetworkTheoreticRegressionModel, kwargs={'k_neighbors': 10, 'alpha_tvd': 0.01})

combi = knnr_meta + kmcr_meta + dtr_meta + ntr_meta

# sampleSizes = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
sampleSizes = [500]
resampleCount = 5

mtf = ModelTestFramework()

# mtf.timeModels(
#     modelMetas=combi,
#     data=wells_merged_clean_sample,
#     sampleSizes=sampleSizes,
#     resampleCount=resampleCount
# )

# Test baseline with country field inclded
mtf.timeModels(
    modelMetas=ca_meta,
    data=wells_merged_clean_sample,
    sampleSizes=sampleSizes,
    resampleCount=resampleCount,
    xCols=['lon', 'lat', 'country'],
    yCol='tvd'
)
