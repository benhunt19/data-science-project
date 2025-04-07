from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED, TEST_DATA_FOLDER, WELLS_MERGED_TEST, TEST_RESULTS_FOLDER
from src.models.networkTheoreticRegressionModel import NetworkTheoreticRegressionModel
from src.testing.modelMetaMaker import ModelMetaMaker as MMM
from datetime import datetime

wells_merged = pd.read_csv(f'{DATA_FOLDER}/{WELLS_MERGED}.csv')

wells_merged_clean = wells_merged[['lat', 'lon', 'tvd','country']].copy()
wells_merged_clean = wells_merged_clean[wells_merged_clean['tvd'] > 0].dropna(subset=['lat', 'lon', 'tvd'])

sample_size = len(wells_merged_clean)
train_pcnt = 0.80

wells_merged_clean_sample = wells_merged_clean.sample(sample_size)


train_df = wells_merged_clean_sample.iloc[:int(sample_size*train_pcnt), :]
test_df = wells_merged_clean_sample.iloc[int(sample_size*train_pcnt) :, :]


ntr_meta = MMM.createMeta(
    model=NetworkTheoreticRegressionModel,
    kwargs={
        'k_neighbors': 10,
        'alpha_tvd': 0.01
    }
)

mtf = ModelTestFramework()

mtf.testModels(
    modelMetas=ntr_meta,
    x_train=train_df[['lat', 'lon']],
    y_train=train_df['tvd'],
    x_test=test_df[['lat', 'lon']],
    y_test=test_df['tvd']
)

mtf.evaluateResults()

mtf.saveMetrics(filename=f'ntr_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}')