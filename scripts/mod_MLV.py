
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
from mod_main import list_profile_DFs

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

# Method: add nitrate difference across mld
def add_crossML_grad(df_glid, mlstat_glid, bounds = [10,30]):
    """ 
    Add the gradient across the ML by taking the nitrate difference
    
    """
    mlstat_glid['nitrate_cross'] = np.tile(np.nan, mlstat_glid.shape[0])
    mlstat_glid['nitrate_underML'] = np.tile(np.nan, mlstat_glid.shape[0])
    validcross=[]
    novalid = []

    for ind, prof in enumerate(dfproc.list_profile_DFs(df_glid)):
        mld = mlstat_glid.loc[ind, 'mld']
        lowerprof = prof[(prof.pressure>mld+bounds[0]) & (prof.pressure<mld+bounds[1])]
        # print(len(lowerprof))

        if len(lowerprof) > 0:
            arr = lowerprof.nitrate.values
            presarr = lowerprof.pressure.copy()*10000 # convert to Pa 

            under_sum = np.trapz(arr, presarr )/9.8

            # Note we divide by the actual pressures observed since they do not range perfectly
            presrange = (presarr.max() - presarr.min())/10000 # in m again
            under_mean = under_sum/presrange # in umol/m3


            under_mean = under_mean/1000 # in umol/kg

            # print(under_mean)

            cross = mlstat_glid.loc[ind, 'nitrate_mean'] - under_mean
            mlstat_glid.loc[ind, 'nitrate_cross'] = cross
            mlstat_glid.loc[ind, 'nitrate_underML'] = under_mean

            validcross.append(cross)
            # nitrate_cross.append(sum)
        else:
            mlstat_glid.loc[ind, 'nitrate_cross'] = np.nan
            novalid.append(ind)
            # nitrate_cross.append(sum)
    
    print(str(len(validcross)) + ' out of ' + str(len(mlstat_glid)) + ' profiles')
    # print(novalid)
    
    return mlstat_glid

# %% Variance Calculations

def get_ml_bx(df_plat, dav_plat, rho_ref):
    """ 
    # Fixed version, updated June 04 2024
    # Specify a mean rho from points above MLD
    # Calculate b from around chosen depth d0 (rather than mean b for whole ML)
    # Use center difference for slight smoothing 

    Find horiz buoyancy gradient at chosen depth (in mixed layer)
    @param:     df_plat     glider DataFrame
                dav_plat    dive-averaged glider DF
                rho_ref     reference density for buoyancy calculation
    @return:    depth_bx    DF with Bx values at chosen d0
    """
    result = dav_plat.copy()
    result['b'] = np.tile(np.nan, len(result))

    # Find density within mixed layer for each glider profile
    for ind, prof in enumerate(sg.list_profile_DFs(df_plat)):
        mldepth = dav_plat.mld.iloc[ind]
        profind = int(str(prof.profid.values[0])[-4:])

        if (not np.isnan(mldepth)) & (profind < len(dav_plat)):
            upperprof = prof[(prof.pressure < (mldepth+1))]
            rho = np.nanmean(upperprof.sigma0) + 1000
            # Buoyancy calculation
            result.loc[profind, 'b'] = 9.81 * (1 - (rho/rho_ref) )


    # Add horizontal distance (x, in m) between profiles
    # result = result.reset_index()
    result['xdist'] = np.tile(0, len(result))
    result.loc[1:, 'xdist'] = dist_between_profiles(result)

    # Horizontal buoyancy gradient calculation as deltaB/deltaX
    result['bx'] = np.tile(np.nan, len(result))
    for i in np.arange(1, len(result)-1, 1):
        delta_b = result['b'].iloc[i+1] - result['b'].iloc[i-1]
        delta_x = result['xdist'].iloc[i] + result['xdist'].iloc[i+1]
        result.loc[i, 'bx'] = np.abs(delta_b)/delta_x

    return result

def calc_interML_variability(mldf_glid, var_list):
    """ 
    No longer used -- vertical variance within single profile. 
    """
    profids = mldf_glid.profid.unique()
    variability_DF = pd.DataFrame(columns= (['yearday'] + var_list))

    for ind, prof in enumerate(profids):
        dat = mldf_glid[mldf_glid.profid==prof]
        variability_DF.loc[ind, 'yearday']= dat.yearday.mean()

        for variable in var_list:
            variability_DF.loc[ind, variable] = np.var(dat[variable])

    return variability_DF



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



# %% Outdated:

# VERSION 1: Specify a mean rho from around d0. 
#  Calculate b from around chosen depth d0 (rather than mean b for whole ML)
# Try again with different reference density 

# def get_d0_bx(df_plat, dav_plat, d0 = 50, thresh=20):
#     """ 
#     Find horiz buoyancy gradient at chosen depth (in mixed layer)
#     @param:     df_plat     glider DataFrame
#                 dav_plat    dive-averaged glider DF
#                 d0          depth at which to calculate Bx
#                 thresh      acceptable threshold away from d0
#     @return:    depth_bx    DF with Bx values at chosen d0
#     """
#     depth_bx = pd.DataFrame()
#     for ind, prof in enumerate(sg.list_profile_DFs(df_plat)):
#         # Reference density at depth d0=50, th5
#         # mean of depth_bx at depth d0=50, th5
#         rho_ref = 27.15355581299716 + 1000
#         # rho_ref = 27.59259584413856 +1000

#         # Filter profile DF into rows within depth threshold
#         filtered_df = prof[(prof['pressure'] >= (d0 - thresh)) & (prof['pressure'] <= (d0 + thresh))].copy()
#         filtered_df['b'] = np.tile(np.nan, len(filtered_df))

#         # If we have obs around d0, and we have a potential density
#         if (len(filtered_df)>0) & (not np.isnan(rho_ref)):
#             buoyancy = lambda sigma0: 9.81 * (1 - ((sigma0 + 1000)/rho_ref) )
#             filtered_df.loc[:, 'b'] = buoyancy(df_plat.sigma0)
#             # Find the row with the closest value of "pressure" to chosen d0
#             closest_row = filtered_df.loc[(filtered_df['pressure'] - d0).abs().idxmin()]
#             depth_bx = pd.concat([depth_bx, closest_row], axis=1)

#     depth_bx = depth_bx.T

#     # # Add distance (m) between profiles
#     dist = dist_between_profiles(depth_bx)
#     depth_bx = depth_bx.reset_index()
#     depth_bx['xdist'] = np.tile(0, len(depth_bx))
#     depth_bx.loc[1:, 'xdist'] = dist

#     # # Now take hz difference in B between successive profiles
#     depth_bx['bx'] = np.tile(np.nan, len(depth_bx))
#     for i in np.arange(1, len(depth_bx), 1):
#         B1 = depth_bx.b[i-1]
#         B2 = depth_bx.b[i]
#         diff = np.abs(B2-B1)
#         dist = depth_bx.xdist[i]
#         depth_bx.at[i, 'bx'] = diff/dist

#     return depth_bx

# # Find mean sigma (reference rho) in just mixed layer
# def ml_mean_rho(df_plat):
#     result = []
#     for ind, prof in enumerate(sg.list_profile_DFs(df_plat)):
#         # Find potential density within mixed layer
#         mldepth = dav_plat.iloc[ind].mld.item()
#         if not np.isnan(mldepth):
#             upperprof = prof[(prof.pressure < (mldepth+5))]
#             rho_ref = upperprof.sigma0.mean() + 1000
#         else:
#             rho_ref = np.nan
#         result.append(rho_ref)
#     return np.nanmean(result)

# # Average sigma in ML across all 
# # print(ml_mean_rho(df_659))
# # print(ml_mean_rho(df_660))

# # Average sigma in ML across all 
# print(np.nanmean(df_659.sigma0) + 1000)
# print(np.nanmean(df_660.sigma0) + 1000)

# # Find mean sigma (reference rho) at chosen d0
# def d0_mean_sigma(df_plat, d0=50, thresh=5):
#     result = []
#     for ind, prof in enumerate(sg.list_profile_DFs(df_plat)):
#         filtered_df = prof[(prof['pressure'] >= (d0 - thresh)) & (prof['pressure'] <= (d0 + thresh))].copy()
#         result.append(filtered_df.sigma0.mean())
#     return result

# # reference mean density, from just around chosen d0 
# ls = np.array(d0_mean_sigma(df_plat, d0=50, thresh=5))
# print(np.nanmean(ls))

# # if you want to reference mean of entire water column
# print(np.nanmean(df_plat.sigma0))



    