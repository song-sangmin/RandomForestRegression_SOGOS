"""
Module for functions dealing with Pandas Dataframes
Revised Apr 16 2024
"""


import xarray as xr
import pandas as pd
import numpy as np
import gsw 
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
import importlib as lib
from scipy.io import savemat
import mod_main as sg
from mod_L3proc import vert_profiles
import math
import haversine as hs   
from haversine import Unit





dir = '../data/ACC_fronts/'
PF = pd.read_csv(dir + 'PF.csv', header=None)
SAF = pd.read_csv(dir + 'SAF.csv', header=None)
SIF = pd.read_csv(dir + 'SIF.csv', header=None)
STF = pd.read_csv(dir + 'STF.csv', header=None)
for csv in [PF, SAF, SIF, STF]:
    csv.columns = ['lon', 'lat']

# from mod_main import dav_659, dav_660
# %% Format data from xarray Datasets into Pandas dataframes

def flatten2DF(g3, nandrop=False): 
    """
    Flatten the gridded-pressure (gp) product into a dataframe. 
    Allows you to calculate MLD and run SOML code to estimate nitrate and pH.
    @param: g3 - gridded-pressure gp_659 or gp_660
    """ 
    GliderData = {

        "juld" : g3.time.values.flatten(),
        "yearday" : g3.days.values.flatten(),
        "lat" : g3.lat.values.flatten(),
        "lon" : g3.lon.values.flatten(),
        "pressure" : g3.P.values.flatten(),
        "CT" : g3.CT.values.flatten(),
        "SA" : g3.SA.values.flatten(),
        "oxygen" : g3.oxygen.values.flatten(),
        "buoyancy" : g3.buoyancy.values.flatten(),
        "bbp700" : g3.bbp700.values.flatten(),
        "bbp695" : g3.bbp695.values.flatten(),
        "bbp470" : g3.bbp470.values.flatten(),
    }

    # Build repeat arrays before flattening, to match dimensions of DF
    depths = pd.DataFrame(g3.depth.values)
    depths_array = pd.concat([depths.T]*len(g3.nprof)).T
    GliderData["depth"] = depths_array.values.flatten()
    
    profnum = pd.DataFrame(g3.nprof.values)
    profnum_array = pd.concat([profnum.T]*len(g3.depth)) # no transpose
    GliderData["profid"] = profnum_array.values.flatten()

    divenum = pd.DataFrame(g3.dive.values)
    divenum_array = pd.concat([divenum.T]*len(g3.depth))  # no transpose
    GliderData["dive"] = divenum_array.values.flatten()

    GliderData["sigma0"] = gsw.sigma0(g3.SA.values.flatten(), g3.CT.values.flatten())
    GliderData["spice"] = gsw.spiciness0(g3.SA.values.flatten(), g3.CT.values.flatten())
    DF = pd.DataFrame(GliderData)

    if nandrop:
        # Drop Nans in appropriate columns
        cols_nonan = [col for col in DF.columns if col not in ['buoyancy', 'bbp700', 'bbp695', 'bbp470']] 
        DF = DF.dropna(axis=0, subset=cols_nonan).copy()

    DF = DF.sort_values(by=['profid', 'depth'])

    return DF

def dfvar_to_darray(df, var='pH'):
    """" 
    Convert the dataframe pH variable back into a DataArray so it can be added to the original
    gridded product gp_659 and gp_660. 
    From there, the pH can be interpolated onto the isopycnal gridded gi_659 and gi_660."""
    
    list = []
    nprof = np.arange(0,df.nprof.values.max()+1,1)   # +1 since the index starts at 0
    depth = np.arange(0,1001,1)                     # change this for depth resolution

    for i in nprof:
        profile_df = df[df.nprof==i]
        depths = profile_df.depth.values
        v = profile_df[var].values

        # Column required to match dimensions of 1001
        vert_var = np.empty(1001); vert_var[:] = np.NaN
        for ind, d in enumerate(depths): # put obs at the right depth
            vert_var[d] = v[ind]
        list.append(vert_var)   

    arr = np.array(list).T
    return xr.DataArray(arr, dims = ["depth", "nprof"], 
                    coords = [depth, nprof] )


# %% Basic functions for handling dataframes
def list_profile_DFs(df):
    """ 
    @param df: dataframe with all profiles
    @return: list of dataframes, each with a unique profile
    """
    profids = pd.unique(df.profid)
    profile_DFs = []
    for i in range(len(profids)):
        profile_DFs.append(df[df['profid']==profids[i]].copy())
    return profile_DFs



def make_diveav(df, thresh=0.005, mld_lim=[8,12]):
    """
    Make dive-averaged dataframes (per profile). 
    @param:    df: dataframe with all profiles
               thresh: threshold for finding sigma in each profile
               mld_lim: limits for finding mld
    """
    prof_list = list_profile_DFs(df)

    newDF = pd.DataFrame()
    newDF['profid'] = pd.unique(df.profid)
    newDF['yearday'] = [np.nanmean(x.yearday) for x in prof_list]
    newDF['lat'] = [x.lat.mean() for x in prof_list]
    newDF['lon'] = [x.lon.mean() for x in prof_list]

    pressure=[]
    no10counter = 0
    lenrangesig = []
    nomld=0
    nanprofids = []

    for prof in prof_list:
        prof = prof.sort_values(by='pressure')
        temp10 = prof[(prof.pressure>mld_lim[0]) & (prof.pressure<mld_lim[1])]

        if len(temp10)>0:
            dens10m = np.nanmean(temp10.sigma0.values)
            dens_tofind = dens10m + 0.05 # threshold used in dove 2021

            rangesig = pd.DataFrame()
            rangesig = prof[(prof.sigma0 < (dens_tofind + thresh)) & (prof.sigma0 > (dens_tofind-thresh))].copy()
            
            lenrangesig.append(len(rangesig))
            
            if len(rangesig) == 0:
                nomld=nomld+1
                pres = np.nan
                nanprofids.append(prof.profid.values[0])
            else:
                pres = np.nanmean(rangesig.pressure)

        else: 
            no10counter=no10counter+1
            pres = np.nan
            nanprofids.append(prof.profid.values[0])

        pressure.append(pres)

    newDF['mld'] = pressure
    print('number of profiles with no 10m data: ', no10counter)
    print('approx # of obs within profile that were averaged for final mlp: ', np.mean(lenrangesig))
    print('number of nan mlds even where 10m exists: ', nomld)
    return newDF, nanprofids



def get_track_eke(dav_plat, aviso, buffer=0.04, daily=False):
    """ 
    Get along-track EKE from satellite data, which is averaged between days 120-200
    Use surface glider lat/lon to co-locate. 

    SOGOS BOUNDS: 
    datestart='2019-04-30'
    dateend='2019-07-25'
    dateend='2021-07-25'
    lat1=-56.8; lat2=-43
    lon1=19; lon2=41
    data_sat = aviso.sel(time=slice(datestart, dateend))
    data_sat = data_sat.sel(latitude=slice(lat1, lat2))
    data_sat = data_sat.sel(longitude=slice(lon1, lon2))
    avg_aviso = data_sat.mean(dim='time')

    """
    dates = sg.ytd2datetime(dav_plat.yearday)
    avg_aviso = aviso.mean(dim='time') # if using daily=false

    track_eke = []      # initialize array to return, eke for each day
    for idx, date in enumerate(dates): 
        lx = dav_plat.lat[idx] #.values[0]
        ly = dav_plat.lon[idx] #.values[0]
        glider_lat = [lx-buffer, lx, lx+buffer]  # circular? 
        glider_lon = [ly-buffer, ly, ly+buffer]

        if daily:
            day_aviso = aviso.sel(time=date, method='nearest')
            point = day_aviso.eke.interp(latitude=lx, longitude=ly, method = 'linear')
        else:
            point = avg_aviso.eke.interp(latitude=lx, longitude=ly, method = 'linear')

        track_eke.append(point.values.tolist())
    return track_eke

def get_track_FSLE(diveav, FSLE, buffer=0.04):
    """ 
    Linearly interpolate to daily EKE values along glider path, get maximum value within radius 
    @param <FSLE>      satellite FSLE over deployment period """
    dates = diveav.date
    
    buffered_fsle = np.array([])       # initialize array to return, fsle for each day
    # track_fsle = np.array([])       # initialize array to return

    for idx, date in enumerate(dates): 
        lx = diveav.latitude[idx] #.values[0]
        ly = diveav.longitude[idx] #.values[0]
        glider_lat = [lx-buffer, lx, lx+buffer]  # circular? 
        glider_lon = [ly-buffer, ly, ly+buffer]

        dayFSLE = FSLE.sel(time=date, method='nearest')
        rangeFSLE = dayFSLE.interp(lat=glider_lat, lon=glider_lon, method = 'linear')
        buffered_fsle = np.append(buffered_fsle, np.amin(rangeFSLE.fsle_max).values)

        # ptFSLE = dayFSLE.interp(lat=lx, lon=ly, method = 'linear')
        # track_fsle = np.append(track_fsle, ptFSLE.fsle_max.values)

    return buffered_fsle


# %% Other functions using dataframes

def between_fronts(data, upper_front, lower_front, buffers=[0.5,0.5]):
    """
    Restrict dataframe to observations between fronts of the ACC. 
    """
    valid = []
    upper_front = upper_front.sort_values(by=['lon'])
    lower_front = lower_front.sort_values(by=['lon'])

    for i in range(len(data)):
        platlon = data.iloc[i].lon
        platlat = data.iloc[i].lat

        upper_front_lat = upper_front[upper_front['lon']>platlon].lat.values[0]
        lower_front_lat = lower_front[lower_front['lon']>platlon].lat.values[0]

        if (platlat< (upper_front_lat + buffers[0])) & (platlat>(lower_front_lat - buffers[1])):
            valid.append(True)
        else:
            valid.append(False)
    
    return data[valid]

# %% For isopycnal analysis (cross spectra and wavelet)

def get_isopycnal_signal(platDF, ave_isopycnal, var_thresh=0.01, var_list = ['yearday', 'pressure', 'sigma0', 'nitrate', 'spice']):
    """ 
    @param: prof_list: list of glider DF's, using list_profile_DFs 
            ave_isopycnal: list of isopycnal values to find in each profile
            var_thresh: threshold for finding sigma in each profile
            var_list: which variables to keep track of
    @return: Dictionary object containing along-isopycnal variables. 
    """
    prof_list = list_profile_DFs(platDF)
    dLine = dict.fromkeys(ave_isopycnal)

    for sig in ave_isopycnal:
        temp = pd.DataFrame()

        for prof in prof_list:
            # Find all sigma points that are within that threshold
            rangeDF = pd.DataFrame() 
            rangeDF = prof[(prof['sigma0']< (sig+var_thresh)) & (prof['sigma0'] > (sig-var_thresh))].copy()

            # Choose mean of values
            rowdat = rangeDF[var_list].copy().dropna()
            rowdat = np.mean(rowdat, axis=0) #nanmean avoided if you drop nans above.
            temp = pd.concat([temp, rowdat], axis=1)
            
        temp = temp.T
        dLine[sig] = temp

    return dLine


# %% Assign TS-bin coordinates

def TSbin(df, nbins):
    nobs, bin_temp = np.histogram(df.CT, nbins)
    nobs, bin_sal = np.histogram(df.SA, nbins)   
    return [bin_temp, bin_sal]

def coords_TSbin(df, nbins): 
    """ Manually give bin coordinates to each obs. row in the dataframe
    Allows you to plot more complex quantities in T-S space. 
    """

    [bin_temp, bin_sal] = TSbin(df, nbins)
    # initialize empty rows which will hold new bin coordinates for T and S
    coordT = np.empty(len(df))
    coordS = np.empty(len(df))

    df = df.sort_values(by='CT')
    dfind = []  # find index limits where observations fall into each bin
    for i in range(len(bin_temp)): 
        dfind.append(np.searchsorted(df.CT, bin_temp[i], side='right'))
    for i in range(len(bin_temp)-1):  # notice -1 = nbins
        coordT[dfind[i]:dfind[i+1]] = i
    df['y_temp'] = coordT.astype(int)


    df = df.sort_values(by='SA')
    dfind = []
    for i in range(len(bin_sal)):
        dfind.append(np.searchsorted(df.SA, bin_sal[i], side='right'))
    for i in range(len(bin_sal)-1): 
        coordS[dfind[i]:dfind[i+1]] = i
    df['x_sal'] = coordS.astype(int)

    df = df.sort_values(by=['profid', 'pressure'])
    return df

def array_TSbin(df, nbins, var='oxygen', stat='mean'):
    """" Calculates value for the TS-binned array.
    Note that coordT will be stored as "y_temp" and corresponds to row in the array."""
    arr = [ [np.NaN for i in range(nbins)] for j in range(nbins) ]
    for r in range(nbins):
        for c in range(nbins):
            subdf = df[(df['x_sal']==c) & (df['y_temp']==r)]

            if stat == 'mean':
                arr[r][c] = np.nanmean(subdf[var])
            elif stat == 'variance':
                arr[r][c] = np.var(subdf[var])
            elif stat == 'count':
                arr[r][c] = len(subdf[var])
            # change line here 

    return arr


def TS_contours(df, nbins, sminus=0.23, splus=0.2, tminus=1.5, tplus=0.6, type='density'):
    """ 
    @ return: gridvals  - 2D array of density or spice values
    """
    # Add density contours
    # Figure out boudaries (mins and maxs)
    smin = df.SA.min() - sminus
    smax = df.SA.max() + splus

    tmin= df.CT.min() - tminus
    tmax = df.CT.max() + tplus # 0.5 for df659

    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1+1,0))
    ydim = int(round((tmax-tmin)+1,0))
    
    # Create empty grid of zeros
    gridvals = np.zeros((ydim,xdim))
    
    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin

    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):

            if type == 'density':
                gridvals[j,i]=gsw.sigma0(si[i],ti[j])
            elif type == 'spice':
                gridvals[j,i]=gsw.spiciness0(si[i],ti[j])

    # To plot in figure, use the following line: 
    # CS = ax.contour(si,ti,dens, linestyles='dashed', colors='k', alpha=0.4, zorder=3)

    return si, ti, gridvals

#%% Prepare dataframes for ESPER

def DF2mat(df, filename):
    """
    Converts a dataframe to a .mat file.
    """
    dict = {"temp": df.temperature.values, 
        "sal": df.salinity.values, 
        "oxygen": df.oxygen.values, 
        "depth": df.pressure.values,
        "lat": df.lat.values, 
        "lon": df.lon.values}

    scipy.io.savemat(filename, dict)
    print('Saved to ' + filename)
    return


#%% Add dNdz and AOU

def add_dist2maxb(platDF):
    """
    Calculate buoyancy (actually Nsquared using gsw) 
    @param      argo_DF: dataframe with argo profiles 
                ---> make new variable: profiles (list): list of dataframes, each dataframe is a profile
    @return     list of dataframes, each dataframe is a profile with a buoyancy column added    
    """
    new_DF = pd.DataFrame()

    profids = pd.unique(platDF.profid) # list of profile ids
    profile_DFs = [] # list of dataframes, each corresponding to a float profile

    for i in range(len(profids)):
        profile_DFs.append(platDF[platDF['profid']==profids[i]].copy())

    for profile in profile_DFs:
        b = profile.log_buoyancy.values
        ind = np.where(b == np.nanmax(b))

        if len(b) > 4:   # five points in profile or more
            depth_maxb = profile.pressure.values[ind].max()
        else:
            depth_maxb = np.nan       

        profile['dist_maxb'] = np.tile(depth_maxb,len(b)) - profile.pressure.values
        
        new_DF = pd.concat([new_DF, profile])
        # To add buoyancy to each profile, use the code line:
        # argo_buoy = add_Pchip_buoyancy(argo_qc)  

    return new_DF

def add_nitrategrad(platDF):
    """
    Calculate nitrate gradient dNO3/dZ using derivative of Pchip interpolator 
    @param      platDF: dataframe with argo profiles 
    @return     list of dataframes, each dataframe is a profile with a dN/dZ column added    
    """
    new_DF = pd.DataFrame()

    # profids = pd.unique(platDF.profid) # list of profile ids
    # profile_DFs = []
    # for i in range(len(profids)):
    #     profile_DFs.append(platDF[platDF['profid'] == profids[i]].copy())

    profile_DFs = list_profile_DFs(platDF)

    for profile in profile_DFs:

        if np.isnan(profile.nitrate).all():
            nans = np.empty(len(profile['pressure'])); nans[:] = np.NaN
            profile.loc[:, 'grad_nitrate'] = nans
        else:
            f = scipy.interpolate.PchipInterpolator(x=profile.pressure.values, y=profile.nitrate.values, extrapolate = False)
            devf = f.derivative()
            grad_nitrate = devf(profile["pressure"].values)
            # 
            
            profile.loc[:, 'grad_nitrate'] = grad_nitrate
        
        # Take vert N2 and find the maximum in the profile. 
        
        new_DF = pd.concat([new_DF, profile])

    return new_DF

def add_Pchip_buoyancy(plat_DF):
    """
    Calculate buoyancy (actually Nsquared using gsw) 
    @param      plat_DF: dataframe with profiles 
                ---> make new variable: profiles (list): list of dataframes, each dataframe is a profile
    @return     list of dataframes, each dataframe is a profile with a buoyancy column added    
    
    Version 09.06.2023
    """
    new_DF = pd.DataFrame()

    profids = pd.unique(plat_DF.profid) # list of profile ids
    profile_DFs = []
    for i in range(len(profids)):
        profile_DFs.append(plat_DF[plat_DF['profid'] == profids[i]].copy())

    for profile in profile_DFs:
        Nsquared, mid_pres = gsw.Nsquared(profile.SA.values, profile.CT.values, 
                                            profile.pressure.values, profile.lat.values)

        df = pd.DataFrame.from_dict({"Ns": Nsquared, "mp": mid_pres})
        df = df.dropna()

        if np.isnan(df.mp).all():
            nans = np.empty(len(profile['P'])); nans[:] = np.NaN
            profile.loc[:, 'buoyancy'] = nans
        else:
            f = scipy.interpolate.PchipInterpolator(x=df.mp, y=df.Ns, extrapolate = False)
            vertN2 = f(profile["pressure"].values)

            surf = np.where(~np.isnan(profile.SA.values))[0][0]
            bottom = np.where(~np.isnan(profile.SA.values))[0][-1]
            vertN2[surf] = vertN2[surf+1]
            vertN2[bottom] = vertN2[bottom-1]
            profile.loc[:, 'buoyancy'] = vertN2
        
        # Take vert N2 and find the maximum in the profile. 
        
        new_DF = pd.concat([new_DF, profile])

    return new_DF


def daily_dist(dav):
    temp = dav.yearday.round()
    cind = list(np.arange(0,len(temp)-1,1))

    la1 = dav.lat.loc[cind[:-1]]
    lo1 = dav.lon.loc[cind[:-1]]
    la2 = dav.lat.loc[cind[1:]]
    lo2 = dav.lon.loc[cind[1:]]

    # make pairs
    result = []

    for ind in cind[:-1]:
        loc1=(la1.iloc[ind], lo1.iloc[ind])
        loc2=(la2.iloc[ind], lo2.iloc[ind])
        result.append(hs.haversine(loc1,loc2,unit=Unit.METERS))
    
    return result


