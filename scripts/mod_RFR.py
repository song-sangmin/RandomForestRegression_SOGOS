import pandas as pd
import xarray as xr
import numpy as np
import scipy
import math
from scipy.stats import iqr


var_list = {
            'Model_A': ['spice', 'sigma0'],
            'Model_B': ['CT', 'SA', 'pressure'], 
            'Model_C': ["CT", "SA", "pressure", 'oxygen'],
            'Model_D': ["CT", "SA", "pressure", "oxygen", 'lat', 'lon', 'yearday'],
            'Model_E': ["CT", "SA", 'pressure', 'oxygen', 'lat', 'lon', 'yearday', 'o2sat'],
            'Model_F': ["CT", "SA", 'pressure', 'oxygen', 'lat', 'lon', 'yearday', 'log_buoyancy'],
            'Model_G': ["CT", "SA", 'pressure', 'oxygen', 'lat', 'lon', 'yearday', 'ydcos', 'ydsin']}

model_list = list(var_list.keys())

# ML data
dir = '../working-vars/RF-training/'
mltraining = pd.read_csv(dir + 'mldata_training.csv')
mlvalidation = pd.read_csv(dir + 'mldata_validation.csv')
mltest = pd.read_csv(dir + 'mldata_testing.csv')

# Download results from the RF training
dir = '../working-vars/RF-training/'
# Import data output from RF training
RF_validation = dict.fromkeys(model_list)
RF_test = dict.fromkeys(model_list)
RF_modelmetrics = pd.read_csv(dir + 'mlresult_model_metrics_full.csv')
RF_featimps = pd.read_csv(dir + 'mlresult_feat_imps_full.csv').set_index('Mdl')

# Use RF_test in place of RF_ver[4]
# Use RF_validation in place of RF_ver[5]
for modtag in model_list:
    filename = 'mlresult_test_err_' + modtag[-1] + '_full.csv'
    RF_test[modtag] = pd.read_csv(dir + filename)
    filename = 'mlresult_val_err_' + modtag[-1] + '_full.csv'
    RF_validation[modtag] = pd.read_csv(dir + filename)

# Download cross validation 
dir = '../working-vars/RF-crossval/'
RF_loo = pd.read_csv(dir + 'loo_metrics_byModel.csv')
RF_loo_WMO = pd.read_csv(dir + 'loo_metrics_byWMO.csv')
RF_kfold = pd.read_csv(dir + 'kfold_metrics_byModel.csv')

# NN Comparisos
RF_pred_6030 = pd.read_csv('../working-vars/RF-training/mlresult_sgfloat_allpreds_full.csv')
ESPER_pred = pd.read_csv('../working-vars/ESPER-prediction/wmo5906030_df_with_esper_dec2023.csv')
CANYON_pred = pd.read_csv('../working-vars/CANYON-prediction/wmo5906030_df_with_canyon_dec2023.csv')

# %% Methods for preparing ML Data
""" 
Includes methods for splitting datasets into training, validation, test
"""

def split_profiles(floatDF, test_split = False):
    """ 
    @param floatDF: scaled dataframe with all float profiles
    @return: 
        training_data: training data (80% of profiles) with ship added
        validation_data: validation data (20% of profiles)
    
    This method treats a float profile like the smallest discrete "unit" upon which to train and validate the model.
    Our goal is to predict a water column (profile) rather than a discrete observation, so profiles are kept together while splitting.
    Note that test data will be all profiles from the SOGOS float when test_split is set as False by default.

    """

    # Create list of unique profile ID's and select random 80% for training
    profs = pd.unique(floatDF.profid)
    if 5906030 in profs:
        print('Warning: SOGOS float is in training data.')

    np.random.seed(42) 
    training_profid = np.random.choice(profs, int(np.floor(0.8*len(profs))), replace=False)
    training_data = floatDF[floatDF['profid'].isin(training_profid)]

    if test_split == False:
        # Take remaining profiles that were not in training set, for validation
        validation_profid = [x for x in profs if x not in training_profid]
        validation_data = floatDF[floatDF['profid'].isin(validation_profid)]
        test_data = [] # will become SOGOS data

    else:
        # Take HALF of remaining profiles that were not in training set, for validation (10% of total)
        profs_vt = [x for x in profs if x not in training_profid]  # remaining profiles after training data removed
        validation_profid= np.random.choice(profs_vt, int(np.floor(0.5*len(profs_vt))), replace=False)
        validation_data = floatDF[floatDF['profid'].isin(validation_profid)]

        test_profid = [x for x in profs_vt if x not in  validation_profid]
        test_data = floatDF[floatDF['profid'].isin(test_profid)]

    return training_data, validation_data, test_data

def split_ship(shipDF, test_split = False):
    """ 
    Split ship data into training and validation sets
    Stations are kept together (by index). 
    @param test_split: whether to split the 20% remaining into 10% test and 10% validation
    """
    ind = shipDF.index.values
    np.random.seed(42)
    
    training_ind = np.random.choice(ind, int(np.floor(0.8*len(ind))), replace=False)
    training_data = shipDF[shipDF.index.isin(training_ind)]

    if test_split == False:
        validation_ind = [x for x in ind if x not in training_ind]
        validation_data = shipDF[shipDF.index.isin(validation_ind)]
        test_data = [] # will become sogos float
    else:
        ind_vt = [x for x in ind if x not in training_ind]
        validation_ind = np.random.choice(ind_vt, int(np.floor(0.5*len(ind_vt))), replace=False)
        validation_data = shipDF[shipDF.index.isin(validation_ind)]
        
        test_ind = [x for x in ind_vt if x not in validation_ind]
        test_data = shipDF[shipDF.index.isin(test_ind)]
    
    return training_data, validation_data, test_data




# %% Ancillary functions
     
def get_95_bounds(data):
    mean = np.mean(data); std_dev = np.std(data)
    low = mean - 2 * std_dev
    high = mean + 2 * std_dev

    return [low, high]

def get_depth_bias(data, ranges, var='val_error'):
    """ Get validation errors in 100m depth bins. """
    # Example range to pass as @param: ranges
    # pressure_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
    #                 (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
    return {f"{start}-{end}": data[(data["pressure"] >= start) & (data["pressure"] < end)][var].values
            for start, end in ranges}

def print_errors(data, var ='test_relative_error', pres_lim= [0,1000]):
    """ 
    @ param: data:  dataset that has "test_relative_error" in it
    """
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

    err = data[data.yearday <210][var]
    [low, high] = get_95_bounds(err)
    print("\nDuring SOGOS between depths " + str(pres_lim[0]) + ' to ' + str(pres_lim[1]) + ':')
    print('95% of errors fall between:')
    print(str(low.round(5)) + ' to ' + str(high.round(5)) )


# %% Classes for storing model results

class ModelVersion: 
    """ 
    Object holding different RF model runs. 
    """
    def __init__(self, ind_list):
        self.ind_list = ind_list
        self.models = {model: None for model in ind_list}
        self.MAE = {model: None for model in ind_list}
        self.IQR = {model: None for model in ind_list}
        self.r2 = {model: None for model in ind_list}
        self.DF_err = {model: None for model in ind_list}
        self.val_err = {model: None for model in ind_list}
    
    # def add(self, model_name):
    #     self.models[model_name] = None
    #     self.MAE[model_name] = None
    #     self.IQR[model_name] = None
    #     self.r2[model_name] = None
    #     self.DF_err[model_name] = None
    #     self.val_err[model_name] = None
    
    def copy(self):
        return self

    
    def get_metrics(self):
        model_metrics = pd.DataFrame()
        model_metrics['ind'] = self.ind_list
        model_metrics['validation_MAE'] = [x[1][1].item() for x in self.MAE.items()]
        model_metrics['validation_IQR'] = [x[1][1].item() for x in self.IQR.items()]
        model_metrics['validation_r2'] = [x[1][1].item() for x in self.r2.items()]
        model_metrics['test_MAE'] = [x[1][2].item() for x in self.MAE.items()]
        model_metrics['test_IQR'] = [x[1][2].item() for x in self.IQR.items()]
        model_metrics['test_r2'] = [x[1][2].item() for x in self.r2.items()]
        model_metrics.set_index('ind', inplace=True)
        
        return model_metrics


# %% Classes for storing cross-validation results
    
# class KFold:
#     """
#     From a single KFold run (using one model list)
#     Object holds the data from all folds
#     (indexed by fold number, e.g. 1, 2, 3...)
#     """
#     def __init__(self, nfolds=10):
#         fold_list = np.arange(1,nfolds+1)
#         self.fold_list = fold_list # model_list, 
#         self.folds = {k: None for k in fold_list} # models
#         self.MAE = {k: None for k in fold_list}
#         self.IQR = {k: None for k in fold_list}
#         self.r2 = {k: None for k in fold_list}
#         self.DF_err = {k: None for k in fold_list}
#         self.val_err = {k: None for k in fold_list}
    
#     def get_metrics(self):
#         folds_metrics = pd.DataFrame()
#         folds_metrics['fold'] = self.fold_list
#         folds_metrics['validation_MAE'] = [x[1][1].item() for x in self.MAE.items()]
#         folds_metrics['validation_IQR'] = [x[1][1].item() for x in self.IQR.items()]
#         folds_metrics['validation_r2'] = [x[1][1].item() for x in self.r2.items()]
#         folds_metrics['test_MAE'] = [x[1][2].item() for x in self.MAE.items()]
#         folds_metrics['test_IQR'] = [x[1][2].item() for x in self.IQR.items()]
#         folds_metrics['test_r2'] = [x[1][2].item() for x in self.r2.items()]
#         folds_metrics.set_index('fold', inplace=True)
        
#         return folds_metrics

class CrossVal_KFold:
    """ 
     Larger object containing CV information across all models in model_list
     (indexed by model, e.g. 'Model_X')
     During k-fold, we combine errors across folds for each model, 
     such that we add 10% of validation errors from each fold,
     (represent 100% of training data)
    """
    def __init__(self, model_list):
        self.model_list = model_list
        self.val_DF = {k: None for k in model_list} # full pd DF

        # Series of all errors, for plotting histograms
        self.val_error = {k: None for k in model_list} # series of all errors
        self.val_relative_error = {k: None for k in model_list}
        # List of k MAEs. Mean/STD are for Table 1: Val Errors
        self.MAEs = {k: None for k in model_list}
        self.IQRs = {k: None for k in model_list}

    def get_metrics(self, model_list=None): 
        """ 
        Calculate metrics.
        Notice different metrics from the ModelVersion class.
        Here, we want to combine data across the 10 folds for each model,
        i.e. aggregate errors across folds, indexed by Model_X"""
        cvk_metrics = pd.DataFrame()
        # Fill in metrics
        if model_list == None:
            model_list = self.model_list
        
        for ind, mdl in enumerate(model_list):
            cvk_metrics.at[ind, 'median_MAEs'] = np.nanmedian(self.MAEs[mdl].values) # median of 10 medians from folds
            cvk_metrics.at[ind, 'mean_MAEs'] = np.nanmean(self.MAEs[mdl].values)
            cvk_metrics.at[ind, 'mean_IQRs'] = iqr(self.IQRs[mdl].values)

            ab_err = np.abs(self.val_error[mdl].values)
            cvk_metrics.at[ind, 'total_median_AE'] = np.nanmedian(ab_err) # median of all combined errors across folds
            cvk_metrics.at[ind, 'total_mean_AE'] = np.nanmean(ab_err)
            cvk_metrics.at[ind, 'total_IQR'] = iqr(ab_err)
            cvk_metrics.at[ind, 'total_median_bias'] = np.nanmedian(self.val_error[mdl].values)

        cvk_metrics['model'] = self.model_list
        cvk_metrics = cvk_metrics.set_index('model') # .fillna(np.nan)

        return cvk_metrics

def split_kfolds(platDF, nfolds = 10):
        """
        @param platDF: scaled dataframe with all float and ship profiles
                nfolds: number of folds to split data into. 
                        Each fold will be used for validation once.
        @return folds_training: list of kfold dataframes for training
                folds_validation: list of kfold dataframes for validation

        """
        profs = pd.unique(platDF.profid)

        if 5906030 in profs:
                print('warning: 5906030 is in platDF')

        # Shuffle float profile ID's
        # Each element of fold_profs is a list of profile ID's belonging to the k-th fold
        np.random.seed(42) 
        rng = np.random.default_rng(); rng.shuffle(profs)
        fold_profs = np.array_split(profs, nfolds)

        # Make a dictionary of dataframes for validation and training. 
        # Each fold (1/nth of the total training data) will be used for validation once.
        # All profiles that are not part of that fold are used for training. 
        training_list = []; validation_list = []
        for i in np.arange(0,nfolds):
                df = platDF[platDF['profid'].isin(fold_profs[i])].copy()
                validation_list.append(df)

                df = platDF[~platDF['profid'].isin(fold_profs[i])].copy()
                training_list.append(df)

                
        folds_validation = {k:v for k, v in zip(np.arange(1,nfolds+1), validation_list)}
        folds_training = {k:v for k, v in zip(np.arange(1,nfolds+1), training_list)}

        return folds_training, folds_validation

def split_loos(wmos, floatDF, shipDF):
    loo_training = dict.fromkeys(wmos)
    loo_validation = dict.fromkeys(wmos)

    for withheld in wmos:
        floatdat = floatDF[(floatDF.wmoid!=withheld) & (floatDF.wmoid!=5906030)]
        loo_training[withheld] = pd.concat([floatdat, shipDF], ignore_index=True)
        loo_validation[withheld] = floatDF[floatDF.wmoid==withheld]

    return loo_training, loo_validation






