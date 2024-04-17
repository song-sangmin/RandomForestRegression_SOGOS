import pandas as pd
import xarray as xr
import numpy as np
import scipy
import math

 
# Ancillary funtions       
def get_95_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 2 * std_dev
    high = mean + 2 * std_dev

    return [low, high]


class ModelVersion: 
    """ 
    Object holding different RF model runs. 
    """
    def __init__(self, model_list):
        self.model_list = model_list
        self.models = {model: None for model in model_list}
        self.MAE = {model: None for model in model_list}
        self.IQR = {model: None for model in model_list}
        self.r2 = {model: None for model in model_list}
        self.DF_err = {model: None for model in model_list}
        self.val_err = {model: None for model in model_list}
    
    def add(self, model_name):
        self.models[model_name] = None
        self.MAE[model_name] = None
        self.IQR[model_name] = None
        self.r2[model_name] = None
        self.DF_err[model_name] = None
        self.val_err[model_name] = None

    def print_errors(self, model, var ='test_relative_error', pres_lim= [0,1000]):
        data = self.DF_err[model]

        data = data[(data.pressure > pres_lim[0]) & (data.pressure < pres_lim[1])]
        err = data[var]
        print('Error metric: ' + var)
        print('Restricted to depths ' + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
        print('median abs error: \t' + str(np.abs(err).median()))
        print('mean abs error \t\t' + str(np.abs(err).mean()))

        # Bounds 95
        [low, high] = get_95_bounds(err)
        print('\n95% of errors fall between:')
        print(str(low.round(5)) + ' to ' + str(high.round(5)) )

        err = data[data.yearday <200][var]
        [low, high] = get_95_bounds(err)
        print("\nDuring SOGOS between depths " + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
        print('95% of errors fall between:')
        print(str(low.round(5)) + ' to ' + str(high.round(5)) )


    def print(self):
        print('hi')


class ModelMetrics:
    """ 
    Create model metrics from a ModelVersion object.
    """
    def __init__(self, ModelVersion):
        MAE= ModelVersion.MAE
        IQR = ModelVersion.IQR
        r2 = ModelVersion.r2

        model_metrics = pd.DataFrame()
        model_metrics['model'] = ModelVersion.model_list
        model_metrics['validation_MAE'] = [x[1][1].item() for x in MAE.items()]
        model_metrics['validation_IQR'] = [x[1][1].item() for x in IQR.items()]
        model_metrics['validation_r2'] = [x[1][1].item() for x in r2.items()]
        model_metrics['test_MAE'] = [x[1][2].item() for x in MAE.items()]
        model_metrics['test_IQR'] = [x[1][2].item() for x in IQR.items()]
        model_metrics['test_r2'] = [x[1][2].item() for x in r2.items()]

        model_metrics.set_index('model', inplace=True)
        self.DF = model_metrics

    def print(self, sorted=True):
        if sorted:
            return self.DF.sort_values(by='validation_MAE')
        else:
            return self.DF
    
    def printhi(self):
        print('hi')