from src.models.modelClassBase import Model
import pandas as pd

class CountryAverageModel(Model):
    def __init__(self):
        super().__init__()
        self.combiDF = None
        self.countryStore = None
    
    def train(self, x_train : pd.DataFrame, y_train : pd.DataFrame):
        self.combiDF = x_train.copy()
        self.combiDF['y_train'] = y_train
        self.countryAverageDF = self.combiDF.groupby('country')['y_train'].mean()
    
    def test(self, x_test : pd.DataFrame, y_test : pd.DataFrame):
        predictions = [self.countryAverageDF[row['country']] for _, row in x_test.iterrows()]
        return predictions

    @staticmethod
    def __model_name__():
        return 'CountryAverage'