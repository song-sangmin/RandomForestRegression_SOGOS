
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
import mod_DFproc as dfproc
from mod_L3proc import vert_profiles


dir = '../working-vars/ML-variability/'
mlstat_float = pd.read_csv( dir +'mixedlayerstat_float_may24.csv')
mlstat_659 = pd.read_csv( dir +'mixedlayerstat_659_may24.csv')
mlstat_660 = pd.read_csv( dir +'mixedlayerstat_660_may24.csv')

hvariance_659 = pd.read_csv(dir + 'hvariance_659_may24.csv')
hvariance_660 = pd.read_csv(dir + 'hvariance_660_may24.csv')

bx100_659 = pd.read_csv('../working-vars/ML-variability/bx100_659.csv')
bx100_660 = pd.read_csv('../working-vars/ML-variability/bx100_660.csv')

mldf_659 = pd.read_csv(dir + 'mldf_659_may24.csv')
mldf_660 = pd.read_csv(dir + 'mldf_660_may24.csv')



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


# %%

def get_baseML_stat(mldf_glid, dav_glid):
    """ 
    Add the gradient across the ML by taking the nitrate difference
    
    """
    baseML_stat = pd.DataFrame(columns=['profid', 'yearday', 'grad_nitrate', 'log_buoyancy'])

    listDFs = dfproc.list_profile_DFs(mldf_glid)
    for ind, dat in enumerate(listDFs):
        # value = dat.iloc[-1][var] # last element of profile, so closest to MLD
        # list.append(value)
        # baseML_stat.at[ind, 'grad_nitrate'] = dat.iloc[-1]['grad_nitrate']
        baseML_stat.at[ind, 'log_buoyancy'] = dat.iloc[-1]['log_buoyancy']
        baseML_stat.at[ind, 'yearday'] = dat.iloc[-1]['yearday']
        baseML_stat.at[ind, 'profid'] = int(dat.iloc[-1]['profid'])

    return baseML_stat



