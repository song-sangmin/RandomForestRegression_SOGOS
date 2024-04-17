"""
Module for Xarray Dataset functions
Level 3 Gridded Dataset Processing (DFproc for dataframes)
Revised Apr 16 2024
"""

# %%

import xarray as xr
import pandas as pd
import numpy as np
import gsw 
import matplotlib.pyplot as plt
from cmocean import cm as cmo
from datetime import datetime
import scipy
# import glidertools as gt
import importlib as lib
import sgmod_main as sg

# /Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/SOGOS/data/sogos_Apr19_L23/sg660_level3.nc
file_path = '/Users/sangminsong/Library/CloudStorage/OneDrive-UW/Code/SOGOS-ML/data/seaglider-APL/'
L3_659 = xr.open_dataset(file_path + 'sg659_level3.nc')
L3_660 = xr.open_dataset(file_path + 'sg660_level3.nc')

# %% Gridded - Pressure 

def simplify_grid(L3):
    """
    Simplify L3 grid by removing unnecessary variables.
    """
    g3 = L3[['time', 'dive', 'lat', 'lon', 'speed', 
             'P', 'SA', 'CT', 'dissolved_oxygen',
             'wlbb2fl_sig695nm_adjusted', 'wlbb2fl_sig470nm_adjusted',
             'wlbb2fl_sig700nm_adjusted']].copy() #700nm to match argo data
                           
    g3["sigma"] = xr.DataArray(data=gsw.sigma0(g3.SA, g3.CT), dims=g3.dims)
    g3['days'] = sg.datetime2ytd(g3.time)

    # 'S':'salinity', 'T':'temperature', 
    g3 = g3.rename({'dissolved_oxygen':'oxygen_raw'})
    g3 = g3.rename({'wlbb2fl_sig700nm_adjusted':'bbp700'})
    g3 = g3.rename({'wlbb2fl_sig695nm_adjusted':'bbp695'})
    g3 = g3.rename({'wlbb2fl_sig470nm_adjusted':'bbp470'})
    
    g3 = g3.rename_dims({'half_profile_data_point':'nprof', 'z_data_point':'depth'})

    g3 = g3.transpose()   # avoids .T everytime you plot
    g3 = g3.assign_coords({'nprof':g3.nprof.values, 'depth':g3.depth.values})


    # drop off end dates from recovery period
    # temp = g3.days.where(g3.days<300, drop=True)
    # g3 = g3.sel(nprof=temp.nprof)
    g3 = trim(g3, min=119, max=250, coord='days')

    # g3 = add_AOU(g3)
    g3 = Pchip_buoyancy(g3)

    return g3

def trim(g3, min=119, max=210, coord='days'):
    """ Can also trim by coord='nprof', good for quick checking data
    """
    #Interesting! Note that you have to do g3.coord.where and not just g3.where
    #Transposing above will make your grid return a weird pickle lock thread error.
    #Fixed below
    temp = g3[coord].where((g3[coord]>min) & (g3[coord]<max), drop=True)
    g3 = g3.sel(nprof=temp.nprof)

    return g3

def vert_profiles(grid):
    """"
    Use on L2 or L3. 
    Returns a list of datasets, each corresponding to an interpolated vertical profile on one glider dive."""
    profiles = [] # initialize a list of datasets (each dataset will be one vertical profile ~ 1 glider dive)
    for idx, n in enumerate(grid.nprof):
        vert = grid.sel(nprof=n)   # vertical slice of the grid (profile over depth for one dive)
        profiles.append(vert)
    return profiles
    
#%% Adding calculated variables

def Pchip_buoyancy(gp):
    profiles = vert_profiles(gp)
    list = [] # each row appended will be a vertical profile.

    for profile in profiles:
            Nsquared, mid_pres = gsw.Nsquared(profile["SA"].values, profile["CT"].values, 
                                                profile["P"].values, profile["lat"].values)

            df = pd.DataFrame.from_dict({"Ns": Nsquared, "mp": mid_pres})
            df = df.dropna()

            if np.isnan(df.mp).all():
                nans = np.empty(len(profile['P']))
                nans[:] = np.NaN
                list.append(nans)
            else:
                f = scipy.interpolate.PchipInterpolator(x=df.mp, y=df.Ns, extrapolate = False)

                vertN2 = f(profile["P"].values)

                surf = np.where(~np.isnan(profile.SA.values))[0][0]
                bottom = np.where(~np.isnan(profile.SA.values))[0][-1]
                vertN2[surf] = vertN2[surf+1]
                vertN2[bottom] = vertN2[bottom-1]

                list.append(vertN2)
    
    arr = np.array(list).T
    gp["buoyancy"] = xr.DataArray(arr, dims = ["depth", "nprof"], 
                        coords = [gp.depth.values, gp.nprof.values] )

    return gp

def add_AOU(g3):
    """
    Add AOU to the grid Dataset. 
    Dataset should already have absolute salinity SA and conservative temperature CT."""
    O2_sol = gsw.O2sol(g3['SA'], g3['CT'], g3['P'], g3['lon'], g3['lat'])
    AOU = O2_sol - g3['oxygen']
    newgrid = xr.merge([g3, AOU.to_dataset(name="AOU")])
    
    return newgrid

# %% Gridding in Density Space 

def prof_dataframe(profile):
    """ 
    Create a pandas DataFrame for a single vertical profile. 
    Vertical profiles come from the vert_profiles() function. 
    All variables of interest are filtered for nans and sorted by sigma,
    to be used before transformation into density space."""

    col = {'sigma' : profile.sigma.values,
              'SA' : profile.SA.values,
              'CT' : profile.CT.values,
              'oxygen' : profile.oxygen.values,
              'oxygen_raw' : profile.oxygen_raw.values,
            #  add corrected oxygen 
            #  add attribute 
              'pressure' : profile.P.values,
            #   'pH' : profile.pH.values,
              'buoyancy' : profile.buoyancy.values}
    df = pd.DataFrame.from_dict(col)

    # filter out nans/repeats and sort by sigma so it is monotonically increasing
    df = df.dropna()
    df = df.drop_duplicates(subset = ['sigma'])
    df = df.sort_values('sigma')

    return df


def transform_dataset(g3, sigma_ref, vars = ['SA', 'CT', 'oxygen', 'P', 'buoyancy']): #, 'pH'
    """
    Transform gridded-pressure into a Dataset in density space.
    Density at reference sigma's are estimated by PChip interpolation.
    DataArray coordinates are (sigma, nprof)
    """

    ds = xr.Dataset()
    ds['time'] = xr.DataArray(g3.time, dims = ["sigma", "nprof"], 
                    coords = [sigma_ref, g3.nprof.values] )
    ds['days'] = xr.DataArray(g3.days, dims = ["sigma", "nprof"],
                    coords = [sigma_ref, g3.nprof.values] )
    ds['lat'] = xr.DataArray(g3.lat, dims = ["sigma", "nprof"], 
                    coords = [sigma_ref, g3.nprof.values] )
    ds['lon'] = xr.DataArray(g3.lon, dims = ["sigma", "nprof"], 
                    coords = [sigma_ref, g3.nprof.values] )
    # ds['P'] =  xr.DataArray(g3.P, dims = ["sigma", "nprof"], 
    #                 coords = [sigma_ref, g3.nprof.values] )
    # ds['mld'] =  xr.DataArray(g3.mld, dims = ['nprof'], 
    #                 coords = [g3.nprof.values] )
        

    ds['dive'] = xr.DataArray(g3.dive, dims = ["nprof"],
                    coords = [g3.nprof.values] )

    df_list = [prof_dataframe(profile) for profile in vert_profiles(g3)]

    for var in vars: 
        list = [] # each row in list is an interpolated vertical profile 

        for df in df_list:
            sigma_obs = df.sigma.values 
            obs = df[var].values

            # 660 has profiles with only nans, so add list of nans instead of interpolating
            if np.isnan(df.sigma).all() or np.isnan(df[var]).all() :
                nans = np.empty(len(sigma_ref))
                nans[:] = np.NaN
                list.append(nans)
            else:
                f = scipy.interpolate.PchipInterpolator(x=sigma_obs, y=obs, extrapolate=False)
                list.append(f(sigma_ref))
        arr = np.array(list).T
        ds[var] = xr.DataArray(arr, dims = ["sigma", "nprof"], 
                    coords = [sigma_ref, g3.nprof.values] )
    
    ds = add_AOU(ds)
    return ds

def transform_var(gp, sigma_ref, var):
    """
    Return a DataArray of a single variable in density space.
        Density at reference sigma's are estimated by PChip interpolation.
        DataArray coordinates are (sigma, nprof) to match original gi dataset.
    """    
    df_list = [prof_dataframe(profile) for profile in vert_profiles(gp)]
    list = [] # each row in list is an interpolated vertical profile 

    # for n, df in enumerate(df_list):
    # sigma_obs = df.sigma.values 
    # obs = df[var].values

    # if ~np.isfinite(obs).all():
    #     print('not finite. nprof = ', str(n), ' ind = ', str(np.where(~np.isfinite(obs))))
    #     print ('values: ', obs[np.where(~np.isfinite(obs))])

        
    for df in df_list:
        sigma_obs = df.sigma.values 
        obs = df[var].values

        # 660 has profiles with only nans, so add list of nans instead of interpolating
        if np.isnan(df.sigma).all() or np.isnan(df[var]).all() :
            nans = np.empty(len(sigma_ref))
            nans[:] = np.NaN
            list.append(nans)
        else:
            f = scipy.interpolate.PchipInterpolator(x=sigma_obs, y=obs, extrapolate=False)
            list.append(f(sigma_ref))

    arr = np.array(list).T
    darray = xr.DataArray(arr, dims = ["sigma", "nprof"], 
                coords = [sigma_ref, gp.nprof.values] )

    return darray


# iso_659 = np.linspace(26.8,27.8,1001)  
# gi_659 = gproc.transform_dataset(gp_659, iso_659, vars = ['SA', 'CT', 'oxygen', 'oxygen_raw', 'P', 'buoyancy']) 

# iso_660 = np.linspace(26.9,28.0,1001)
# gi_660 = gproc.transform_dataset(gp_660, iso_660,  vars = ['SA', 'CT', 'oxygen', 'oxygen_raw', 'P', 'buoyancy'])


# %% Dive-averaged grid

def simplify_dive_grid(L3, gp):
    dav = L3[['u_dive', 'v_dive', 'surface_curr_east', 'surface_curr_north',
                    'lat_dive', 'lon_dive']]
    dav = dav.rename({'surface_curr_east':'u_surface', 'surface_curr_north':'v_surface',
                            'lat_dive':'lat', 'lon_dive':'lon'})
    dav = dav.rename_dims({'dive_data_point':'dive'})

    # drop off end profiles
    dav = dav.sel(dive=slice(0,gp.dive.values.max()+1))

    # add days as dimension
    t = gp[['dive', 'days']].groupby('dive').mean()
    t = t.mean(dim='depth')
    row = np.empty(len(dav.dive))
    row[:] = np.NaN
    for ind, coord in enumerate(t.dive.values):
        row[coord] = t.days.values[ind]
    dav['yearday'] = xr.DataArray(row, dims=['dive'])

    return dav

def calc_bfsle(dav, FSLE, buffer=0.05):
    """ 
    Linearly interpolate to daily FSLE values along glider path, get maximum value within radius
    Modified to use L3 dav dataset.
    @param <FSLE>      satellite FSLE over deployment period """
    dates = sg.ytd2datetime(dav.days)
    
    buffered_fsle = np.array([])       # initialize array to return, fsle for each dive

    for idx, date in enumerate(dates): 
        lx = dav.lat[idx] #.values[0]
        ly = dav.lon[idx] #.values[0]

        if buffer>0:
            glider_lat = [lx-buffer, lx, lx+buffer]  # circular? 
            glider_lon = [ly-buffer, ly, ly+buffer]
        elif buffer==0:
            glider_lat = lx
            glider_lon = ly

        dayFSLE = FSLE.sel(time=date, method='nearest')
        rangeFSLE = dayFSLE.interp(lat=glider_lat, lon=glider_lon, method = 'linear')
        buffered_fsle = np.append(buffered_fsle, np.amin(rangeFSLE.fsle_max).values)

    return buffered_fsle

def add_dive_bfsle(dav, FSLE, buffer=0.05):
    bfsle = calc_bfsle(dav, FSLE, buffer)
    dav['bfsle'] = xr.DataArray(bfsle, dims=['dive'])
    return dav

# %% Prepare dataframes for optode correction 

def g3_optode_lag(g3):
    profiles = vert_profiles(g3)
    list=[]

    for profile in profiles:
        ind = ~np.isnan(profile.days.values)
        ytd_secs = profile.days.values[ind]*24*60*60
        depth = profile.depth.values[ind]

        secs = ytd_secs - ytd_secs[0]
        secs_row= np.empty(1001)
        secs_row[:] = np.nan

        for i, z in enumerate(depth):
            secs_row[z] = secs[i]

        list.append(secs_row)

    arr = np.array(list).T
    g3['secs'] = xr.DataArray(arr, dims = ["depth", "nprof"], 
                        coords = [g3.depth.values, g3.nprof.values] )
    return g3[['secs', 'oxygen','CT']]