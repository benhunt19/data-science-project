from scipy.stats import wilcoxon

from src.testing.modelTestFramework import ModelTestFramework
from src.globals import DATA_FOLDER, WELLS_MERGED, TEST_DATA_FOLDER, WELLS_MERGED_TEST, TEST_RESULTS_FOLDER
from src.models.countryAverageModel import CountryAverageModel
from src.models.knnRegressionModel import KNNRegressionModel
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


sample_size = 100_000
train_pcnt = 0.8

df = wells_merged_clean.sample(sample_size).reset_index(drop=True)

train_df = df.iloc[:int(sample_size*train_pcnt), :]
test_df = df.iloc[int(sample_size*train_pcnt) :, :]

# Baseline
ca_meta = MMM.createMeta(model=CountryAverageModel, kwargs={})

# Models
knnr_meta = MMM.createMeta(model=KNNRegressionModel, kwargs={'k': 7, 'weights': 'distance'})
kmcr_meta = MMM.createMeta(model=KMeansRegressionModel, kwargs={'k': 100, 'model_type': 'regression'})
ntr_meta = MMM.createMeta(model=NetworkTheoreticRegressionModel, kwargs={'k_neighbors': 10, 'alpha_tvd': 0.01})

combi = ntr_meta + knnr_meta

mtf = ModelTestFramework()
mtf.testModels(
    modelMetas=combi,
    x_train=train_df[['lat', 'lon']],
    y_train=train_df['tvd'],
    x_test=test_df[['lat', 'lon']],
    y_test=test_df['tvd'],
)

mtf.results.to_csv(f'{TEST_RESULTS_FOLDER}/significanceTest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

# mtf.results = pd.read_csv(f'{TEST_RESULTS_FOLDER}/significanceTest_20250407_105712.csv')

# Placeholder: Replace with actual per-sample errors
knnr_errors = abs(mtf.results[mtf.results.columns[1]] - mtf.results[mtf.results.columns[0]])
ntr_errors = abs(mtf.results[mtf.results.columns[2]] - mtf.results[mtf.results.columns[0]])
differences = ntr_errors - knnr_errors

stat, p_value = wilcoxon(differences, alternative='greater')
print(f"p-value: {p_value:.10e}")
print(f"stat: {stat}")