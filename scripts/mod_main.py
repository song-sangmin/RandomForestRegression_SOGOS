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
sg659 = pd.read_csv('../data/glider/sg659_tsO2corr.csv', index_col=0)
sg660 = pd.read_csv('../data/glider/sg660_tsO2corr.csv', index_col=0)

# Glider data
# df_659 = pd.read_csv('../data/glider/mldata_sg659.csv')
# df_660 = pd.read_csv('../data/glider/mldata_sg660.csv')
df_659 = pd.read_csv('../working-vars/RF-glider/mlpred_sg659_G.csv', index_col=0) # fixed
df_660 = pd.read_csv('../working-vars/RF-glider/mlpred_sg660_G.csv', index_col=0)
dav_659 = pd.read_csv('../data/glider/dav_659_EKEPAR.csv', index_col=0)  
dav_660 = pd.read_csv('../data/glider/dav_660_EKEPAR.csv', index_col=0)

# Float data: 
floatDF = pd.read_csv('../data/bgc-argo/mldata_floatDF_qc.csv', index_col=0)
sgfloat = floatDF[(floatDF.yearday<205) & (floatDF.wmoid==5906030)]
# Full float 6030, for long time series MLD
dav_6030 = pd.read_csv('../data/bgc-argo/dav_full6030_noqc.csv', index_col=0) # over all years
dav_float = pd.read_csv('../data/bgc-argo/dav_sgfloat_EKEPAR.csv', index_col=0) # only sogos deployment

# Ship data:
shipDF = pd.read_csv('../data/go-ship/mldata_shipDF_qc.csv') 

# Satellite data:
altimetry = xr.open_dataset('../data/satellite/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1714604183615.nc')
altimetry['eke'] = 0.5*np.sqrt(altimetry.ugosa**2 + altimetry.vgosa**2)

FSLE =  xr.open_dataset('../data/satellite/FSLE_sogos.nc')

# # EKE data bounds
# datestart='2019-04-30'
# dateend='2019-07-25'
# dateend='2021-07-25'
# lat1=-56.8; lat2=-43
# lon1=19; lon2=41
# data_sat = aviso.sel(time=slice(datestart, dateend))
# data_sat = data_sat.sel(latitude=slice(lat1, lat2))
# data_sat = data_sat.sel(longitude=slice(lon1, lon2))


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

# def scale_features(df, training, type='StandardScaler'):
#     """ Scale down glider dataset.
#     @param: df: dataframe to scale
#             type: type of scaler to use, 'StandardScaler' or 'MinMaxScaler'
#     """

#     cols_nonan = [col for col in df.columns if col not in ['profid', 'wmoid']] # all columns to remove nans
#     newdf = df.dropna(axis=0, subset=cols_nonan).copy()
#     new_training = training.dropna(axis=0, subset=cols_nonan).copy()

#     # Drop NaNs *EXCEPT* in the wmoid column, or else ship data is removed

#     if type == 'StandardScaler':
#         sca = preprocessing.StandardScaler().fit(new_training[cols_nonan])

#     elif type == 'MinMaxScaler':
#         sca = preprocessing.MinMaxScaler().fit(new_training[cols_nonan])

#     df_scaled =  pd.DataFrame(sca.transform(newdf[cols_nonan]), columns=newdf[cols_nonan].columns)
#     df_scaled['profid'] = newdf['profid']
#     df_scaled['wmoid'] = newdf['wmoid']

#     return df_scaled


# def scale_down_tvt(training, validation, test, type='StandardScaler'):
#     """ 
#     Scale features using 
#     @param: df: dataframe to scale
#             type: type of scaler to use, 'StandardScaler' or 'MinMaxScaler'
#     """
#     # Don't remove NaNs in wmoid since ship will have empty field
#     cols_nonan = [col for col in training.columns if col not in ['profid', 'wmoid']] # all columns to remove nans

#     new_training = training.dropna(axis=0, subset=cols_nonan).copy()
#     new_validation = validation.dropna(axis=0, subset=cols_nonan).copy()
#     new_test = test.dropna(axis=0, subset=cols_nonan).copy()

 
#     # Scaler is built *USING TRAINING DATA* and applied to all
#     if type == 'StandardScaler':
#         sca = preprocessing.StandardScaler().fit(new_training[cols_nonan])

#     elif type == 'MinMaxScaler':
#         sca = preprocessing.MinMaxScaler().fit(new_training[cols_nonan])

#     # Scale down all using the training scaler
#     training_scaled = pd.DataFrame(sca.transform(new_training[cols_nonan]), columns=new_training[cols_nonan].columns)
#     validation_scaled = pd.DataFrame(sca.transform(new_validation[cols_nonan]), columns=new_validation[cols_nonan].columns)
#     test_scaled = pd.DataFrame(sca.transform(new_test[cols_nonan]), columns=new_test[cols_nonan].columns)

#     # Add back profid, wmoids
#     training_scaled['profid'] = new_training['profid']
#     validation_scaled['profid'] = new_validation['profid']
#     test_scaled['profid'] = new_test['profid']

#     training_scaled['wmoid'] = new_training['wmoid']
#     validation_scaled['wmoid'] = new_validation['wmoid']
#     test_scaled['wmoid'] = new_test['wmoid']

#     return training_scaled, validation_scaled, test_scaled

# def rescale_target(scaled_pred, unscaled_obs, type='Standard Scaler'):
#     "Rescale values to the original range of the data"
#     unscaled_obs =np.array(unscaled_obs).reshape(-1,1)
#     scaled_pred = np.array(scaled_pred).reshape(-1,1)

#     if type == 'StandardScaler':
#         scaler = preprocessing.StandardScaler().fit(unscaled_obs)
#     elif type == 'MinMaxScaler':
#         scaler = preprocessing.MinMaxScaler().fit(unscaled_obs)

#     temp = scaler.inverse_transform(scaled_pred)
#     rescaled = [y.item() for y in temp]
 
#     return rescaled

