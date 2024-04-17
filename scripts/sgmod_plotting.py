# from re import L
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

size = 32
params = {'legend.fontsize': size, 
          'xtick.labelsize':size, 
          'ytick.labelsize':size, 
          'font.size':size,
          'font.family':'Futura'}
plt.rcParams.update(params)

# plt.rcParams['font.size'] = '12'
# plt.rcParams['font.family'] = 'Futura'

lims = {"SA" : [33.5, 35],
       "CT" : [1, 3.5],
       "oxygen" : [150, 350],
       "oxy_tcorr" : [150, 350],
    #    "Chl" : [0, 0.6],
       "AOU" : [10, 120],
       "pH" : [7.63, 8.0],
       "TA" : [2280, 2387],}

palettes = {'SA': cmo.haline, 
        'CT': cmo.thermal, 
        'oxygen': cmo.ice, 
        'oxy_tcorr': cmo.ice, 
        # 'Chl': cmo.algae,
        'AOU': cmo.amp, 
        'nitrate': cmo.deep, 
        'TA': cmo.tempo}

#%% Isopycnals

def get_isopycnals(list_DFs, min=26.8, max=27.8, spacing = 0.2, threshold=0.1):
    
    min = min*10 # 26.84855129122479
    max = max*10 # 27.7599400854142
    isos = np.arange(min, max, spacing*10)/10   # use /10 to get single decimal 

    iso_dict = dict.fromkeys(isos)

    for val in isos:
        isoline = pd.DataFrame()

        for df in list_DFs:
            closest_row = df.loc[(df['sigma0'] - val).abs().idxmin()]
            isoline = pd.concat([isoline, closest_row], axis=1)

        isoline = isoline.T
        isoline = isoline[abs(isoline['sigma0'] - val) <= threshold]
        iso_dict[val] = isoline
    
    # Use the following to plot: 

    # ax2 = ax1.twinx()
    # for ind, (key, val) in enumerate(iso_dict.items()):
    #     # Skipping lines for dates
    #     # Sort the DataFrame by 'yearday' to ensure the data points are in order
    #     val = val.sort_values(by='yearday')

    #     # Initialize variables to store the x and y values for the plot
    #     x_values = []; y_values = []

    #     # Iterate over the DataFrame and build the x and y values with NaN separators
    #     for i in range(len(val) - 1):
    #         x_values.append(val.iloc[i]['yearday'])
    #         y_values.append(val.iloc[i]['pressure'])
    #         if val.iloc[i + 1]['yearday'] - val.iloc[i]['yearday'] > 2:
    #             x_values.append(None)
    #             y_values.append(None)

    #     # Add the last data point
    #     x_values.append(val.iloc[-1]['yearday'])
    #     y_values.append(val.iloc[-1]['pressure'])

    #     ax2.plot(x_values, y_values, linewidth=1, label=key, color=palette[ind])

    return iso_dict

# %% Plotting functions

def plot_var(g3_glider, var='SA'):
    """"
    Quick plot for single variable, for troubleshooting code.
    @param      gp_glider: gridded dataset
                var: Options are 'SA', 'CT', 'oxygen', 'AOU', 'spice'"""
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    g3_glider[var].plot(ax=ax, cmap=cmo.dense)

    ax.margins(x=0.01)
    ax.invert_yaxis()
    ax.set_xlabel('profile number')


def gridx_nprof(g3, vars = ['SA', 'CT'], tag='', save=False, lim=[]):
    """
    @g3: Pass either gridded dataset gp_ on pressure, or gi_ on isopycnals.
    @vars: Options are ['salinity', 'temperature', 'oxygen', 'AOU', 'pH', 'TA']
    """
    
    for v in vars:
        fig = plt.figure(figsize=(10,5))
        ax = fig.gca()

        if len(lim)==0:
            min =  lims[v][0]
            max =  lims[v][1]
        else:
            min = lim[0]
            max = lim[1]
        c = palettes[v]

        g3[v].plot(vmin=min, vmax=max, cmap=c, ax=ax)

        # if v=='Chl':
        #     ax.set_ylim([1026.8, 1027.3])

        ax.margins(x=0.01)
        ax.invert_yaxis()
        ax.set_title(v + ' ' + tag)
        ax.set_xlabel('profile number')

    # if coord is sigma add tag to title

        pngtitle = tag + '_' + v + '_nprof.png'
        if save:
            plt.savefig('figures/' + pngtitle, format='png')

    return

def gridx_days(gp, ycoord='depth', vars = ['SA', 'CT'], tag='', save=False, lim=[]):
    """
    @gp: Pass gridded dataset gp_ on pressure
    @vars: Options are ['salinity', 'temperature', 'oxygen', 'AOU', 'pH', 'TA']
    @ycoord: Options are 'depth', 'sigma'
    """

    for v in vars:
        fig = plt.figure(figsize=(10,5))
        ax = fig.gca()

        if len(lim)==0:
            min =  lims[v][0]
            max =  lims[v][1]
        else:
            min = lim[0]
            max = lim[1]
        
        c = palettes[v]
        
        gp.plot.scatter('days', ycoord, hue=v, vmin=min, vmax=max, cmap=c, ax=ax, s=3, zorder=3)

        ax.margins(x=0.01)
        ax.invert_yaxis()
        ax.set_xlabel('days')
        ax.set_ylabel(ycoord)
        ax.set_title(v + ' ' + tag)
        plt.grid(visible=True, axis='x', zorder=0)

        # ax.set_xticks(np.arange(120,206,5))
        # for label in ax.xaxis.get_ticklabels()[::5]:
        #     label.set_visible(False)

        if ycoord=='depth':
            tg = 'gp' + tag
        elif ycoord=='sigma':
            tg = 'gi' + tag 
        pngtitle = tg + '_' + v + '_days.png'
        if save:
            plt.savefig('figures/' + pngtitle, format='png')

    return

# %%  REFERENCE FLOAT MAPS

# Reference: Full matchup map for floats, so we can see where the holdout floats are
def print_map(floatDF, wmoids, wmo_colors):
    fig = plt.figure(figsize=(14,8))
    ax=plt.gca()

    for wmo in wmoids[wmoids!=5906030]:
        ax.plot(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.2, linewidth=5, label=str(wmo)[3:], zorder=3)
        ax.scatter(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.1, s=20, zorder=3)

    for wmo in [5906030]:
        ax.plot(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.4, linewidth=7, label=str(wmo)[3:], zorder=3)
        ax.scatter(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.1, s=30, zorder=3)

    plt.title('SOGOS BGC-Argo Matchups (2017-2021)')
    rect = patch.Rectangle((30,-54),10,4, fill=True, color="orange", alpha=0.25,linewidth=2, zorder=1)


    # ax.legend()
    ax.add_patch(rect)
    ax.set_xlim(5,81)
    # ax.set_ylim(-60,-30)
    # ax.set_aspect('equal')

    ax.yaxis.set_major_formatter("{x:1.0f}°S")
    ax.xaxis.set_major_formatter("{x:1.0f}°E")

    ax.grid(linestyle='dashed', alpha=0.6, zorder=1)

    lines = ax.get_lines()
    labelLines(lines, align=False, fontsize=18, zorder=3)
# labelLines(lins) # , align=False, fontsize=18)
# plt.plot(shipDF.lon, shipDF.lat, alpha=0.8, linestyle='dashed', c='k', linewidth=5)


# %% FSLE plotting
# """ 
# Plot absolute dynamic topography from daily averages
# """
# def plot_ADT(adt, datestr):
#     plt.figure(figsize=(12, 8))
#     adt.sel(time=datestr).plot(cmap=cmocean.cm.matter)

#     # Add rectangle for SWIR study region
#     rect = patch.Rectangle((20, -55), 20, 10,
#                            fill=False, color="black", linewidth=1)
#     plt.gca().add_patch(rect)


# # Plot deployment period +/- one month, weekly, over entire larger region
# # for date in adt.time[::7]:
#     # plot_ADT(adt,date)

# for date in adt_sogos.time:
#     plot_ADT(adt_sogos, date)

#     # Plot glider path
#     plt.plot(sg659_locs['longitude'],
#              sg659_locs['latitude'], 1, color="yellow")
#     plt.plot(sg660_locs['longitude'], sg660_locs['latitude'], 1, color="white")



# def plot_FSLE(data, datestr):
#     plt.figure(figsize=(12, 8))
#     data.sel(time=datestr).plot(cmap=cmocean.cm.tempo)

#     # Add rectangle for SWIR study region
#     rect = patch.Rectangle((20, -55), 20, 10,
#                            fill=False, color="black", linewidth=1)
#     plt.gca().add_patch(rect)


# for date in fsle_sogos.time:  # over eddy dates
#     plot_FSLE(fsle_sogos, date)

#     # Plot glider path
#     plt.plot(sg659_locs['longitude'], sg659_locs['latitude'], 1, color="white")
#     plt.plot(sg660_locs['longitude'],
#              sg660_locs['latitude'], 1, color="yellow")



