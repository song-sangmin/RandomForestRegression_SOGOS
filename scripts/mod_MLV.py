
import xarray as xr
import pandas as pd
import numpy as np
import gsw 
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
import importlib as lib
from scipy.io import savemat
import haversine as hs   
from haversine import Unit


import mod_main as sg
import mod_DFproc as dfproc
from mod_L3proc import vert_profiles


# %% Download results from mixed layer calculations
dir = '../working-vars/ML-variability/'
mlstat_float = pd.read_csv( dir +'mixedlayerstat_float_may24.csv')
mlstat_659 = pd.read_csv( dir +'mixedlayerstat_659_may24.csv')
mlstat_660 = pd.read_csv( dir +'mixedlayerstat_660_may24.csv')

hvariance_659 = pd.read_csv(dir + 'hvariance_659_may24.csv')
hvariance_660 = pd.read_csv(dir + 'hvariance_660_may24.csv')

bx100_659 = pd.read_csv('../working-vars/ML-variability/bx100_659.csv', index_col=0)
bx100_660 = pd.read_csv('../working-vars/ML-variability/bx100_660.csv', index_col=0)

mldf_659 = pd.read_csv(dir + 'mldf_659_may24.csv')
mldf_660 = pd.read_csv(dir + 'mldf_660_may24.csv')

# Wavelet results
dir = '../working-vars/ML-wavelet/'
d0_660 = {27.3: 174.62, 27.4: 223.69, 27.5: 296.81, 27.6: 396.96}
isos = [27.3, 27.4, 27.5, 27.6]

nitscal_xtime = pd.read_csv(dir + 'nitscal660_xtime.csv', index_col=0)
nitscal_freq = pd.read_csv(dir + 'nitscal660_freq.csv', index_col=0)
nitscal_coi = pd.read_csv(dir + 'nitscal660_coi.csv', index_col=0)
nit_zamp = {k:v for k,v in zip(isos, np.tile(None, 4))}
nit_sigamp = {k:v for k,v in zip(isos, np.tile(None, 4))}

for sig in isos:
    nit_zamp[sig] = pd.read_csv(dir + 'nitscal660_zamp_' + str(int(sig*10)) + '.csv', index_col=0).values
    nit_sigamp[sig] = pd.read_csv(dir + 'nitscal660_signif_amp_' + str(int(sig*10)) + '.csv', index_col=0).values




# %% Mixed layer functions
def make_mldf(df_plat, dav_plat):
    """ 
    Make a dataframe of only the upper mixed layer values.
    @param:    df_plat:   dataframe of plater data
                dav_plat:  dataframe of dive-averaged data
    @return:    mldf: dataframe of plater data with only upper mixed layer data
                mlstats: add on new variables to dav
    """
    mldf_plat = pd.DataFrame()

    for i, prof in enumerate(list_profile_DFs(df_plat)):
        # values that are the same for whole profile
        mlp = dav_plat.mld.iloc[i]

        if mlp != np.nan:
            # Cut off all the data below the mixed layer
            # Create an array containing mld for each row
            upperprof = prof[prof.pressure<=mlp].copy()
            mlp_array = np.tile(mlp, upperprof.shape[0])
            upperprof['mld'] = mlp_array

            # Make dataframe with only the ML values
            mldf_plat = pd.concat([mldf_plat, upperprof], ignore_index=True)

    return mldf_plat


def integrate_ML_var(profML, variable='nitrate'):
    """ 
    Integrate the variable within the mixed layer. See AIES paper for more information.
    @param  profML = dataframe of only ML values from single profile"""

    # To avoid too much extrapolation, we set the surface value to the first value in the ML
    # Also set the base ML value to the closest (deepest ML value)
    profML = profML.sort_values(by='pressure')
    new_row = profML.iloc[0, :].copy()
    new_row['pressure'] = 0
    profML = pd.concat([profML, new_row.to_frame().T])

    profML = profML.sort_values(by='pressure')
    end_row = profML.iloc[-1, :].copy()
    end_row['pressure'] = profML.mld.iloc[0]
    profML = pd.concat([profML, end_row.to_frame().T])

    # Make variable array to integrate
    arr = profML[variable]
    pres_arr = profML.pressure.copy()
    pres_arr = pres_arr*10000 # convert to Pa 

    # Integrate variable over ML using trap rule
    # This uses the hydrostatic relation to calculate integral
    total = np.trapz(arr, pres_arr)/9.8  # in umol/m2 
    avg = total/profML.mld.iloc[0]  # in umol/m2/m 
    avg = avg/1000  # in umol/kg

    return total, avg

def add_dav_integrated(mldf_plat, dav_plat, variable='nitrate'):
    """
    Function to add dive-averaged, integrated values to the dataframe.
    @param: mldf_plat: mixed layer dataframe from plater
            dav_plat: dive-averaged plater dataframe
    """
    mlstats = dav_plat.copy()
    total_name = variable + '_total'
    mean_name = variable + '_mean'

    mlstats[total_name] = np.tile(np.nan, mlstats.shape[0])
    mlstats[mean_name] = np.tile(np.nan, mlstats.shape[0])

    for i, profnum in enumerate(dav_plat.profid.values): # i = profid
        profML = mldf_plat[mldf_plat.profid == profnum].copy()

        if profML.shape[0]>0:  # if no ML data, then nans are kept as is.
            # if there are ML data, add stats to dav_plat
            # Need to use integral function
            # For variance, make sure you take log of bbp values before
            # For total and mean you do this in the intergrate_ML method 
            # arr = profML[variable]
            # if variable == 'bbp470':
            #     arr = np.log(profML[variable])

            mlstats.loc[i,total_name] = integrate_ML_var(profML, variable=variable)[0]
            mlstats.loc[i,mean_name] = integrate_ML_var(profML, variable=variable)[1]

    return mlstats

def add_ML_integrated(mldf_plat, dav_plat, variable='nitrate'):
    
    mlstats = dav_plat.copy()
    total_name = variable + '_total'
    mean_name = variable + '_mean'

    mlstats[total_name] = np.tile(np.nan, mlstats.shape[0])
    mlstats[mean_name] = np.tile(np.nan, mlstats.shape[0])

    mlstats= mlstats.reset_index(drop=True) # need this line for FLOATS
    # for i, profML in enumerate(sg.list_profile_DFs(mldf_plat)):
    for i, profnum in enumerate(dav_plat.profid.values): # i = profid
        profML = mldf_plat[mldf_plat.profid == profnum].copy()

        if len(profML)>0:  # if no ML data, then nans are kept as is.
            # if there are ML data, add stats to dav_plat
            # Need to use integral function

            # For variance, make sure you take log of bbp values before
            # For total and mean you do this in the intergrate_ML method 
            arr = profML[variable]
            if variable == 'bbp470':
                arr = np.log(profML[variable])

            mlstats.loc[i,total_name] = integrate_ML_var(profML, variable=variable)[0]
            mlstats.loc[i,mean_name] = integrate_ML_var(profML, variable=variable)[1]

    return mlstats

# %% Base of mixed layer calculations
def get_baseML_stat(mldf_plat, dav_plat):
    """ 
    Add the gradient across the ML by taking the nitrate difference
    
    """
    baseML_stat = pd.DataFrame(columns=['profid', 'yearday', 'grad_nitrate', 'log_buoyancy'])

    listDFs = dfproc.list_profile_DFs(mldf_plat)
    for ind, dat in enumerate(listDFs):
        # value = dat.iloc[-1][var] # last element of profile, so closest to MLD
        # list.append(value)
        # baseML_stat.at[ind, 'grad_nitrate'] = dat.iloc[-1]['grad_nitrate']
        baseML_stat.at[ind, 'log_buoyancy'] = dat.iloc[-1]['log_buoyancy']
        baseML_stat.at[ind, 'yearday'] = dat.iloc[-1]['yearday']
        baseML_stat.at[ind, 'profid'] = int(dat.iloc[-1]['profid'])

    return baseML_stat

# %% Ancillary functions

def dist_between_profiles(dav):
    temp = [k.round() for k in dav.yearday.values]
    cind = list(np.arange(0,len(temp),1))

    la1 = dav.lat.loc[cind[:-1]] # should this be iloc?
    lo1 = dav.lon.loc[cind[:-1]]
    la2 = dav.lat.loc[cind[1:]]
    lo2 = dav.lon.loc[cind[1:]]

    # make pairs
    result = []

    for ind in np.arange(0, len(lo1), 1):
        loc1=(la1.iloc[ind], lo1.iloc[ind])
        loc2=(la2.iloc[ind], lo2.iloc[ind])
        result.append(hs.haversine(loc1,loc2,unit=Unit.METERS))
    return result

