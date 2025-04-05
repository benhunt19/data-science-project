from src.models.modelClassBase import Model
from src.models.kMeansRegressionModel import KMeansRegressionModel # for demo
import copy
from pprint import pprint

class ModelMetaMaker:
    def __init__(self):
        pass
    
    @staticmethod
    def createMeta(model: Model, kwargs: dict[list]) -> list[dict]:
        """
        Description:
            This creates arrays of model metadata, this is multi dimensional so the output can be LARGE.
        Exaple usage:
            kwargs = {
                'clusters': range(50, 100, 5),
                'clusters2': range(50, 100, 5),
                'model_type': 'regression'
            }
            modelMetas = ModelMetaMaker.createMeta(model=KMeansRegressionModel, kwargs=kwargs)
            this will return 5 x 5 = 25 model metas with the following kwargs:
            {
                'clusters': 50,
                'clusters2': 50,
                'model_type': 'regression'
            },
            {
                'clusters': 50,
                'clusters2': 55,
                'model_type': 'regression'
            },
            ...
            {
                'clusters': 95,
                'clusters2': 95,
                'model_type': 'regression'
            },
            
            etc.
            
        """
        
        base = {
            'model': model,
            'kwargs': {}
        }
        modelMetas = [copy.deepcopy(base)]
        
        for key, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                new_metas = []
                
                # For each existing meta, create variations with the new parameter
                for meta in modelMetas:
                    for v in value:
                        new_meta = copy.deepcopy(meta)
                        new_meta['kwargs'][key] = v
                        new_metas.append(new_meta)
                        
                modelMetas = new_metas
            else:
                # Non iterable, just add the value    
                for meta in modelMetas:
                    meta['kwargs'][key] = value

        return modelMetas

if __name__ == '__main__':
    
    mmm = ModelMetaMaker()
    kwargs = {
        'clusters': range(50, 100, 5),
        'clusters2': range(50, 100, 5),
        'model_type': 'regression'
    }
    mmm.createMeta(model=KMeansRegressionModel, kwargs=kwargs)
    
    