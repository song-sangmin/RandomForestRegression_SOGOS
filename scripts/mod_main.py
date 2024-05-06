# SOGOS Module

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from cmocean import cm as cmo
from datetime import datetime
import seaborn
import scipy
import geopy.distance

import mod_DFproc as dfproc

# %% Data


# To get all MLD's on not-QC data, use: 
# sg659 = pd.read_csv('../data/glider/df_659_tsO2corr_nonandrop_0131.csv')
# sg660 = pd.read_csv('../data/glider/df_660_tsO2corr_nonandrop_0131.csv')

# Input glider data, which has already been quality controlled. 
sg659 = pd.read_csv('../data/glider/sg659_tsO2corr.csv')
sg660 = pd.read_csv('../data/glider/sg660_tsO2corr.csv')

# Glider data
# df_659 = pd.read_csv('../data/glider/mldata_sg659.csv')
# df_660 = pd.read_csv('../data/glider/mldata_sg660.csv')
df_659 = pd.read_csv('../working-vars/RF-glider/mlpred_sg659_G.csv') # fixed
df_660 = pd.read_csv('../working-vars/RF-glider/mlpred_sg660_G.csv')
# good for sure, but use EKE -- se Data Log
# dav_659 = pd.read_csv('../data/glider/dav_659_may24.csv')  
# dav_660 = pd.read_csv('../data/glider/dav_660_may24.csv')
dav_659 = pd.read_csv('../data/glider/dav_659_EKEPAR.csv')  
dav_660 = pd.read_csv('../data/glider/dav_660_EKEPAR.csv')

# Float data: 
floatDF = pd.read_csv('../data/bgc-argo/mldata_floatDF_qc.csv')
sgfloat = floatDF[(floatDF.yearday<205) & (floatDF.wmoid==5906030)]
# Full float 6030, for long time series MLD
dav_6030 = pd.read_csv('../data/bgc-argo/dav_full6030_noqc.csv')
# good for sure, but use EKE -- se Data Log
# dav_float = dav_6030[dav_6030.yearday<210]
dav_float = pd.read_csv('../data/bgc-argo/dav_sgfloat_EKEPAR.csv')

# Ship data:
shipDF = pd.read_csv('../data/go-ship/mldata_shipDF_qc.csv') 

# # Satellite data:
altimetry = xr.open_dataset('../data/satellite/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1714604183615.nc')
altimetry['eke'] = np.sqrt(altimetry.ugosa**2 + altimetry.vgosa**2)

# # EKE data
# datestart='2019-04-30'
# dateend='2019-07-25'
# dateend='2021-07-25'
# lat1=-56.8; lat2=-43
# lon1=19; lon2=41
# data_sat = altimetry.sel(time=slice(datestart, dateend))
# data_sat = data_sat.sel(latitude=slice(lat1, lat2))
# data_sat = data_sat.sel(longitude=slice(lon1, lon2))

# dav_659['eke'] = dfproc.get_track_eke(dav_659, data_sat, daily=True)
# dav_660['eke']= dfproc.get_track_eke(dav_660, data_sat, daily=True)
# # dav_659['eke_avg'] = dfproc.get_track_eke(dav_659, data_sat, daily=False)
# # dav_660['eke_avg']= dfproc.get_track_eke(dav_660, data_sat, daily=False)
# dav_float['eke'] = dfproc.get_track_eke(dav_660, data_sat, daily=True)



# %% Common functions for operating on arrays 

def datetime2ytd(time):
    """" Return time in YTD format from datetime format."""
    return (time - np.datetime64('2019-01-01'))/np.timedelta64(1, 'D')

def ytd2datetime(num):
    """" Return datetime format to YTD in 2019."""
    return (num * np.timedelta64(1,'D')) + np.datetime64('2019-01-01')

def get_ydsines(yearday):
    """ For adding seasonal variable in Training_RandomForest.ipynb"""
    if (yearday < 0) & (yearday > -365):
        yearday = 365+yearday
    if yearday < -365:
        yearday = 365*2 + yearday
    if yearday >= 365:
        yearday = yearday % 365
    ydcos = np.cos(2*np.pi*np.array(yearday)/365)
    ydsin = np.sin(2*np.pi*np.array(yearday)/365)

    return [ydcos, ydsin]

# %% Add trimming functions

def slice_sogos(ds, datestart='2017-04-30', dateend='2019-07-25',
                lat1=-62, lat2=-48,
                lon1=5, lon2=65):
    """ Restrict to glider deployment region, restrict to eddy dates """
    ds_sogos = ds.sel(time=slice(datestart, dateend))
    
    # Smaller zoom in to low-EKE
    ds_sogos = ds_sogos.sel(lat=slice(lat1, lat2))
    ds_sogos = ds_sogos.sel(lon=slice(lon1, lon2))
    return ds_sogos


# %% Add simple printing functions

# Method to check MAE dictionaries
def print_dict(dict):
    print("\n".join("{}  \t{}".format(k, v) for k, v in dict.items()))

def print_sorted(dict):
    print("\n".join("{}  \t{}".format(k, v) for k, v in sorted(dict.items(), key=lambda x:x[1]))) 
    

# %% Add information on vertical resolution

def list_profile_DFs(platdf):
    """ 
    COPIED FROM DFPROC
    @param df: dataframe with all profiles
    @return: list of dataframes, each with a unique profile
    """
    profids = pd.unique(platdf.profid)
    profile_DFs = []
    for i in range(len(profids)):
        profile_DFs.append(platdf[platdf['profid']==profids[i]].copy())
    return profile_DFs


def print_vertical_res(platDF):
    plat_vert_res = []
    for l in list_profile_DFs(platDF):
        temp = l.copy().sort_values(by='pressure')
        pres_diff = np.diff(temp.pressure) # array of differences in pressure along 1 profile
        avg_diff = pres_diff.mean() # average difference in pressure along 1 profile
        plat_vert_res.append(avg_diff)

        fin = np.array(plat_vert_res).mean()
    print('Average vertical resolution within profile (m): \t' + str(fin))

def print_horizontal_res(platDF):
    dat_DF = list_profile_DFs(platDF)
    distances = []
    for ind, df in enumerate(dat_DF[:-1]):
        first = (dat_DF[ind].lat.mean(), dat_DF[ind].lon.mean())
        next = (dat_DF[ind+1].lat.mean(), dat_DF[ind+1].lon.mean())
        dist = geopy.distance.distance(first, next).km
        distances.append(dist)
        fin = np.array(distances).mean()
    print('Average horizontal distance between surfacing (km): \t' + str(fin))

def print_time_res(platDF):
    plat_time_res = []
    plat_prof_times = []

    for l in list_profile_DFs(platDF):
        plat_prof_times.append(l.yearday.mean())

    avg_diff = np.diff(plat_prof_times).mean()
    print('Average time resolution between profiles (days): \t' + str(avg_diff))

def print_profile_nobs(platDF):
    proflens = []
    for l in list_profile_DFs(platDF):
        if (l.pressure.max()>900) & (l.pressure.min()<10):
            proflens.append(len(l.pressure))
    print('Average # obs in a profile reaching >900m: \t' + str(np.mean(proflens)))


# %% Add functions for scaling features

def scale_features(df, training, type='StandardScaler'):
    """ Scale down glider dataset.
    @param: df: dataframe to scale
            type: type of scaler to use, 'StandardScaler' or 'MinMaxScaler'
    """

    cols_nonan = [col for col in df.columns if col not in ['profid', 'wmoid']] # all columns to remove nans
    newdf = df.dropna(axis=0, subset=cols_nonan).copy()
    new_training = training.dropna(axis=0, subset=cols_nonan).copy()

    # Drop NaNs *EXCEPT* in the wmoid column, or else ship data is removed

    if type == 'StandardScaler':
        sca = preprocessing.StandardScaler().fit(new_training[cols_nonan])

    elif type == 'MinMaxScaler':
        sca = preprocessing.MinMaxScaler().fit(new_training[cols_nonan])

    df_scaled =  pd.DataFrame(sca.transform(newdf[cols_nonan]), columns=newdf[cols_nonan].columns)
    df_scaled['profid'] = newdf['profid']
    df_scaled['wmoid'] = newdf['wmoid']

    return df_scaled


def scale_down_tvt(training, validation, test, type='StandardScaler'):
    """ 
    Scale features using 
    @param: df: dataframe to scale
            type: type of scaler to use, 'StandardScaler' or 'MinMaxScaler'
    """
    # Don't remove NaNs in wmoid since ship will have empty field
    cols_nonan = [col for col in training.columns if col not in ['profid', 'wmoid']] # all columns to remove nans

    new_training = training.dropna(axis=0, subset=cols_nonan).copy()
    new_validation = validation.dropna(axis=0, subset=cols_nonan).copy()
    new_test = test.dropna(axis=0, subset=cols_nonan).copy()

 
    # Scaler is built *USING TRAINING DATA* and applied to all
    if type == 'StandardScaler':
        sca = preprocessing.StandardScaler().fit(new_training[cols_nonan])

    elif type == 'MinMaxScaler':
        sca = preprocessing.MinMaxScaler().fit(new_training[cols_nonan])

    # Scale down all using the training scaler
    training_scaled = pd.DataFrame(sca.transform(new_training[cols_nonan]), columns=new_training[cols_nonan].columns)
    validation_scaled = pd.DataFrame(sca.transform(new_validation[cols_nonan]), columns=new_validation[cols_nonan].columns)
    test_scaled = pd.DataFrame(sca.transform(new_test[cols_nonan]), columns=new_test[cols_nonan].columns)

    # Add back profid, wmoids
    training_scaled['profid'] = new_training['profid']
    validation_scaled['profid'] = new_validation['profid']
    test_scaled['profid'] = new_test['profid']

    training_scaled['wmoid'] = new_training['wmoid']
    validation_scaled['wmoid'] = new_validation['wmoid']
    test_scaled['wmoid'] = new_test['wmoid']

    return training_scaled, validation_scaled, test_scaled

def rescale_target(scaled_pred, unscaled_obs, type='Standard Scaler'):
    "Rescale values to the original range of the data"
    unscaled_obs =np.array(unscaled_obs).reshape(-1,1)
    scaled_pred = np.array(scaled_pred).reshape(-1,1)

    if type == 'StandardScaler':
        scaler = preprocessing.StandardScaler().fit(unscaled_obs)
    elif type == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler().fit(unscaled_obs)

    temp = scaler.inverse_transform(scaled_pred)
    rescaled = [y.item() for y in temp]
 
    return rescaled


# # 3.0.2 Train RF model
# var_predict = 'nitrate'

# def train_RF(var_list, training, validation, test, ntrees=1000, max_feats = 'sqrt', scaler='MinMaxScaler', scale_target = True):
#     """ 
#     Main method to train the RF model.
#     Scaling of datasets to between 0 and 1 is done internally within the method
#     @param: 
#         var_list: list of variables to use in the model
#         training: training data unscaled, i.e. original range of values
#         validation: validation data unscaled
#         test: test data unscaled 
#         ntrees: 1000 trees by default.

#     @return:
#         Mdl: trained RF model
#         Mdl_MAE: Rescaled mean absolute error for training, validation, and test sets
#         Mdl_IQR: Rescaled IQR for training, validation, and test sets
#         DF_with_error: Rescaled dataframe with error metrics for the *TEST* set
#     """

#     Mdl = RandomForestRegressor(ntrees, max_features = max_feats, random_state = 0, bootstrap=False)
#         #  max_features: use at most X features at each split (m~sqrt(total features)). ISSR.
#         #  bootstrapping: *** Should be FALSE for geospatial data.

#     # Drop NaN's without profid or wmoid
#     cols_na = [col for col in training.columns if col not in ['profid', 'wmoid']]
#     training_nona = training.dropna(axis=0, subset=cols_na)  # makes same length as training_scaled
#     validation_nona = validation.dropna(axis=0, subset=cols_na)
#     test_nona = test.dropna(axis=0, subset=cols_na)

#     # Scale features using the specified 'type' for each subset of data
#     # Method drops Nan's
#     [training_scaled, validation_scaled, test_scaled] = scale_down_tvt(training_nona, validation, test, type=scaler)

#     # if old way is correct:
#     # training_scaled = scale_features(training, type=scaler)
#     # validation_scaled = scale_features(validation, type=scaler)
#     # test_scaled = scale_features(test, type=scaler)

#     # Create X Variables for each subset of data. Scale down. 
#     X_training = training_scaled[var_list].to_numpy()
#     X_validation = validation_scaled[var_list].to_numpy()
#     X_test = test_scaled[var_list].to_numpy()

#     if scale_target == True:
#         Y_training = training_scaled[var_predict].to_numpy()
#         Y_validation = validation_scaled[var_predict].to_numpy()
#         Y_test = test_scaled[var_predict].to_numpy()    ### Can also leave target nitrate unscaled. change test_scaled to test
#     else:
#         Y_training = training_nona[var_predict].to_numpy()
#         Y_validation = validation_nona[var_predict].to_numpy()
#         Y_test = test_nona[var_predict].to_numpy()    ### Can also leave target nitrate unscaled. change test_scaled to test


#     # Train the model
#     Mdl.fit(X_training, Y_training)

#     # Estimate
#     Y_pred_training = Mdl.predict(X_training)
#     Y_pred_validation = Mdl.predict(X_validation)
#     Y_pred_test = Mdl.predict(X_test)

#     if scale_target == True:
#         Y_pred_training = rescale_target(Y_pred_training, training_nona[var_predict], type=scaler) # Rescale the predicted nitrate
#         Y_pred_validation = rescale_target(Y_pred_validation, training_nona[var_predict], type=scaler) # Rescale the predicted nitrate
#         Y_pred_test = rescale_target(Y_pred_test, training_nona[var_predict], type=scaler) # Rescale the predicted nitrate
#     # changed from rescale_var to rescale_target for using only the training scaler. 


#     # Create dataframe for the test set with depth --> 
#     DF_with_error = test_nona.copy(); 
#     DF_with_error = DF_with_error.reset_index(drop=True)
#     observed_nitrate = DF_with_error[var_predict].to_numpy()

#     # Save new dataframe with test results
#     DF_with_error['test_prediction'] = Y_pred_test
#     DF_with_error['test_error'] = DF_with_error['test_prediction'] - observed_nitrate
#     DF_with_error['test_relative_error'] = DF_with_error['test_error']/observed_nitrate

#     # Error metrics
#     AE_RF_training = np.abs(Y_pred_training - training_nona[var_predict])
#     IQR_RF_training = iqr(abs(AE_RF_training))
#     r2_RF_training = r2_score(training_nona[var_predict], Y_pred_training)

#     AE_RF_validation = np.abs(Y_pred_validation - validation_nona[var_predict])
#     IQR_RF_validation = iqr(abs(AE_RF_validation))
#     r2_RF_validation = r2_score(validation_nona[var_predict], Y_pred_validation)

#     AE_RF_test = np.abs(Y_pred_test - test_nona[var_predict])
#     IQR_RF_test = iqr(abs(AE_RF_test))
#     r2_RF_test = r2_score(test_nona[var_predict], Y_pred_test)

#     Mdl_MAE = [np.nanmedian(abs(AE_RF_training)), np.nanmedian(abs(AE_RF_validation)), np.nanmedian(abs(AE_RF_test))]
#     Mdl_IQR = [IQR_RF_training, IQR_RF_validation, IQR_RF_test]
#     Mdl_r2 = [r2_RF_training, r2_RF_validation, r2_RF_test]

#     return [Mdl, Mdl_MAE, Mdl_IQR, Mdl_r2, DF_with_error]

