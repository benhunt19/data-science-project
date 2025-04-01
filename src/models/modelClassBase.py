class Model:
    """
    Description:
        Base parent class for use with ML models
    """
    def __init__(self):
        self.model = None

    def train(self):
        pass
    
    def test(self):
        pass
    
    def plot(self):
        print('No plot defined for this class')

    @staticmethod
    def __model_name__():
        return 'BaseModel'