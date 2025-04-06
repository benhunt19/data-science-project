from src.testing.modelTestFramework import ModelTestFramework
import pandas as pd
from src.globals import DATA_FOLDER, WELLS_MERGED, TEST_DATA_FOLDER, WELLS_MERGED_TEST, TEST_RESULTS_FOLDER
from src.models.decisionTreeRegressionModel import DecisionTreeRegressionModel
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


dtr_meta = MMM.createMeta(
    model=DecisionTreeRegressionModel,
    kwargs={
        'maxDepth': 15,
        'minSamplesSplit': 15
    }
)

mtf = ModelTestFramework()

mtf.testModels(
    modelMetas=dtr_meta,
    x_train=train_df[['lat', 'lon']],
    y_train=train_df['tvd'],
    x_test=test_df[['lat', 'lon']],
    y_test=test_df['tvd']
)

mtf.evaluateResults()

mtf.saveMetrics(filename=f'dtr_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}')