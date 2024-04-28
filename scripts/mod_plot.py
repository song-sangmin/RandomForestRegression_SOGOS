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
import mod_main as sg
import seaborn as sns


def my_params(size=16):
    """ 
    Use: plt.rcParams.update(sgplot.prms(size=16))
    """
    plt.style.use('default')
    params = {'legend.fontsize': size, 
            'xtick.labelsize':size, 
            'ytick.labelsize':size, 
            'font.size':size,
            'font.family':'Futura',
            'mathtext.fontset':'stixsans',
            'mathtext.bf':'STIXGeneral:bold'}
    return params

# %% Color Palettes

# Make Model palettes (mod April 2024)
model_palettes = {}
model_palettes['Model_A'] = sns.color_palette('Purples')[3] #change for figures
model_palettes['Model_B'] = sns.color_palette('Reds')[4]
model_palettes['Model_C'] = sns.color_palette('Set2')[4]
model_palettes['Model_D'] = sns.color_palette('Blues')[4]
model_palettes['Model_E'] = sns.color_palette('Set2')[1] 
model_palettes['Model_F'] = sns.color_palette('Set2')[0]
model_palettes['Model_G'] = sns.color_palette('RdPu')[5]

# Set up WMO color palette
wmo_colors = {}
wmo_colors[5904469]=sns.color_palette("RdPu")[4]
wmo_colors[5904659]=sns.color_palette("Purples")[4]
wmo_colors[5905368]=sns.color_palette("Paired")[1]
wmo_colors[5905996]=sns.color_palette("YlGn")[3]
wmo_colors[5906030]='k' # sns.color_palette("Reds")[3]  # SOGOS 
wmo_colors[5906031]=sns.color_palette("Blues")[2]
wmo_colors[5906207]=sns.color_palette("Reds")[4]
wmo_colors[5906007]= sns.color_palette("YlOrRd")[3]

# Set glider palettes
plat_colors = dict({'sg659':'#33BBEE', 'sg660':'#EE3377', 'float':'#CCBB44'})
eke_colors = dict({'high':'#BB5566', 'low':'#004488'})

# Set up units
umol_unit = (r'$\mathbf{[\mu} \mathregular{mol~kg} \mathbf{^{-1}]}$')
# umol_unit = (r'$[\mu \mathregular{mol~kg^{-1}}]$')
eke_unit = (r'$\mathbf{[m^2~s^{-2}]}$')

sigma_unit = (r'$\mathbf{\sigma_0}$ ' + '[kg' + r'$\mathbf{~m^{-3}]}$')
spice_unit = (r'$ \mathbf{\tau} $ ' + '$\mathbf{[m^{-3}~}$' + 'kg]')

# Data

from mod_main import df_659, df_660, dav_659, dav_660, sgfloat 
from mod_RFR import RF_validation, RF_test, RF_modelmetrics, RF_featimps
from mod_RFR import RF_kfold, RF_loo

# %% RANDOM FOREST TRAINING




# %% RANDOM FOREST VALIDATION



# %% RANDOM FOREST TESTING

def compare_resolution(df_glid = df_660, fsize=(8,6), minday=129, maxday=161, maxpres=750, dateformat=True):
    # Plot Float
    fig, axs = plt.subplots(2,1, figsize=fsize, layout='constrained', sharex=True)

    for ax in axs[0:1]:
        dat = sgfloat[(sgfloat.yearday <maxday) & (sgfloat.yearday > minday)]
        sca = ax.scatter(dat.yearday, dat.pressure, c=dat.nitrate, cmap=cmo.deep, s=80, marker='s' , vmin=24.1, vmax=35.9) #, marker='s')
        ax.set_title('BGC-Argo Float Nitrate', fontsize=16)

    for ax in axs[1:]:
        dat = df_glid[(df_glid.yearday <maxday) & (df_glid.yearday > minday)]
        dat = dat[dat.pressure < maxpres]
        sca = ax.scatter(dat.yearday, dat.pressure, c=dat.nitrate_G, cmap=cmo.deep, s=20, vmin=24.1, vmax=35.9) #, vmin=1.5, vmax=3.6) #, marker='s')
    ax.set_title('Glider RFR Nitrate', fontsize=16)
    plt.colorbar(sca, ax=axs[:], shrink=0.5, orientation='vertical').set_label('Nitrate ' + umol_unit, fontsize=16)

    ######################
    ### Always ON

    if dateformat:
        list = np.arange(130, 170, 5)
        list = [sg.ytd2datetime(i) for i in list]
        list = [np.datetime_as_string(i) for i in list]
        list = [i[5:10] for i in list]
        axs[1].set_xticklabels(list)
    else:
        axs[1].set_xlabel('[Yearday]')
    
    for ax in axs:
        ax.set_ylabel('Pressure [dbar]')
        ax.set_xlim([minday, maxday])
        ax.set_ylim([-20, maxpres])
        ax.invert_yaxis()


# %% MIXED LAYER VARIABILITY

def plt_glider_sections(preslim=900, fsize=(12,12), vlist = ['sigma0', 'spice', 'nitrate_G'], font = 14, dateformat=False):
    """
    SG659 and SG660, sections of sigma, spice, nitrate 
    (4 rows, 2 columns)

    @param: pres: pressure range to plot over

    """

    fig, axs = plt.subplots(4,2, figsize=fsize, sharex=True, layout='constrained')
    dotsize = 5

    for ind, df_glid in enumerate([df_659, df_660]):
        df_glid = df_glid[df_glid.pressure<preslim]

        if 'sigma0' in vlist:
            sca = axs[0,ind].scatter(df_glid.yearday, df_glid.pressure,c=df_glid.sigma0, cmap=cmo.dense, s=dotsize, zorder=3)
            plt.colorbar(sca, ax=axs[0,ind], aspect=13).set_label(sigma_unit, fontsize=font)
        if 'spice' in vlist:
            sca2 = axs[1, ind].scatter(df_glid.yearday, df_glid.pressure, c=df_glid.spice, cmap="Oranges", s=dotsize)
            plt.colorbar(sca2, ax=axs[1,ind], aspect=13).set_label(spice_unit, fontsize=font)
        if 'nitrate_G' in vlist:
            sca3 = axs[2, ind].scatter(df_glid.yearday, df_glid.pressure, c=df_glid.nitrate_G, cmap=cmo.deep, s=dotsize, zorder=3, vmin=24.0, vmax=35.8)
            plt.colorbar(sca3, shrink=0.6, ax=axs[2:4,ind], aspect=16).set_label('Nitrate ' + umol_unit, fontsize=font)
            sca4 = axs[3, ind].scatter(df_glid.yearday, df_glid.sigma0, c=df_glid.nitrate_G, cmap=cmo.deep, s=dotsize, zorder=3, vmin=24.0, vmax=35.8)
    
    # Plot left columns, pressure axis 
    for ax in axs[0:3, 0]:
        ax.scatter(dav_659.yearday, dav_659.mld, s=2, c='k', alpha=0.6, zorder=3)
        ax.set_ylabel('Pressure [dbar]')
        ax.set_ylim([-8,preslim+5])

    # Plot right column, pressure axis
    for ax in axs[0:3, 1]:
        ax.scatter(dav_660.yearday, dav_660.mld, s=2, c='k', alpha=0.6, zorder=3)
        ax.set_ylim([-8, preslim+5])
        # ax.set_yticks([])

    # Plot EKE lines and set x range
    for ax in axs.flatten():
        ax.vlines(150, -20, preslim, color='r', linewidth=2, linestyle='dashed', alpha=0.7, zorder=3)
        ax.vlines(170, -20, preslim, color='k', linewidth=2, linestyle='dashed', alpha=0.7, zorder=3)
        ax.set_xlim([120, 200])
        ax.invert_yaxis()

    # Set limits
    for ax in axs[0:3,0]:
        ax.set_ylim([preslim+5,-8])
    for ax in axs[3,:]:
        ax.grid(zorder=1, alpha=0.5)

    # Last row limits
    axs[3,0].set_ylabel(spice_unit)
    axs[3,0].set_ylim([27.87, 26.77])
    axs[3,1].set_ylim([27.99, 26.89])


    # Titles and X labels
    axs[0,0].set_title('SG659', fontsize=font); axs[0,1].set_title('SG660', fontsize=font)
    xt = [120, 140, 160, 180, 200]
    if dateformat: 
        for ax in axs[3,0:2]:
            ax.set_xticklabels(str(sg.ytd2datetime(k))[-5:] for k in xt)
    else:
        for ax in axs[3,0:2]:
            ax.set_xlabel('[Yearday]')

    plt.show()

    return fig, axs


# %% Helpful 


# Set up satellite map plotting
# aviso = xr.open_dataset('../data/satellite/dataset-duacs-nrt-global-merged-allsat-phy-l4_1637011653931.nc')
# aviso['eke'] = np.sqrt(aviso.ugosa**2 + aviso.vgosa**2)

# # Download fronts of the Antarctic Circumpolar Current
# PF = pd.read_csv('../data/ACC_fronts/PF.csv', header=None)
# SIF = pd.read_csv('../data/ACC_fronts/SIF.csv', header=None)
# for csv in [PF, SIF]:
#     csv.columns = ['lon', 'lat']


# %% RFR Plotting


# fig, axs = plt.subplots(2,1, figsize=(11,8), layout='constrained')
# axs = axs.flatten()

# # Plot larger study region
# for ax in [axs[0]]:
#         datestart='2019-04-30'
#         dateend='2019-07-25'
#         lat1=-56.8; lat2=-43
#         lon1=19; lon2=41
#         data_sat = aviso.sel(time=slice(datestart, dateend))
#         data_sat = data_sat.sel(latitude=slice(lat1, lat2))
#         data_sat = data_sat.sel(longitude=slice(lon1, lon2))
#         avg_aviso = data_sat.mean(dim='time')

#         # Plot background EKE (separately because of region area)
#         sca = avg_aviso.eke.plot(ax=ax, cmap='bone_r', alpha=0.8, add_colorbar=False, vmin=0.02, vmax=0.58)
#         # fig.colorbar(sca, ax=ax, shrink=0.5).set_label('EKE')
#         contour = ax.contour(avg_aviso['longitude'], avg_aviso['latitude'], avg_aviso['adt'], 
#                                 colors='k', alpha=0.3, linestyles='solid', zorder=1, levels=4)
#         ax.clabel(contour, inline=True, fontsize=12)  # Add labels to contour lines

#         # Plot float
#         ax.plot(sgfloat.lon,sgfloat.lat, color =  '#CCBB44', alpha=1, linewidth=4, linestyle='dashed',
#                 label='float', zorder=3)  # sns.color_palette("Oranges")[3]
#         ax.scatter(sgfloat.lon,sgfloat.lat, color =  '#CCBB44', alpha=0.9, s=90, 
#                 marker='D', edgecolors='k', zorder=3)

#         # Gliders
#         ax.plot(sg659.lon, sg659.lat, color = plat_colors['sg659'], alpha=0.8, linewidth=6, linestyle='dashed', label='sg659', 
#                 path_effects=[pe.Stroke(linewidth=8, foreground='k'), pe.Normal()], zorder=3) 

#         ax.plot(sg660.lon, sg660.lat, color = plat_colors['sg660'], alpha=0.8, linewidth=4, linestyle='dashed', label='sg660', 
#                 path_effects=[pe.Stroke(linewidth=6, foreground='w'), pe.Normal()], zorder=3) 

# # Plot inset, zoomed into study area with yeardays labeled.
# for ax in [axs[1]]:
#         datestart='2019-04-30'
#         dateend='2019-07-25'
#         lat1=-54.7; lat2=-49.2
#         lon1=29.5; lon2=39.4
#         data_sat = aviso.sel(time=slice(datestart, dateend))
#         data_sat = data_sat.sel(latitude=slice(lat1, lat2))
#         data_sat = data_sat.sel(longitude=slice(lon1, lon2))
#         avg_aviso = data_sat.mean(dim='time')

#         # Plot float
#         ax.plot(dav_float.lon,dav_float.lat, color =  '#CCBB44', alpha=1, linewidth=4, linestyle='dashed',
#         label='float', zorder=3)  # sns.color_palette("Oranges")[3]
#         ax.scatter(dav_float.lon.values,dav_float.lat.values, color =  'k', alpha=0.9, s=140, 
#                 marker='D', edgecolors='k', zorder=3)
#         ax.scatter(dav_float.lon.values,dav_float.lat.values, color =  '#CCBB44', alpha=0.9, s=110, 
#                 marker='D', edgecolors='k', zorder=3)
        
#         for ind, day in enumerate([int(x) for x in dav_float.yearday.values]):
#                 if day in ([120,135,140,150, 160, 170, 185, 205]):
#                         ax.text(dav_float.lon.loc[ind]-.3, dav_float.lat.loc[ind]-.4, str(day), 
#                                         fontsize=16, color='k',
#                         path_effects=[pe.Stroke(linewidth=5, foreground='#CCBB44'), pe.Normal()], zorder=3)

#         # Plot Glider
#         ax.plot(sg660.lon, sg660.lat, color = plat_colors['sg660'], alpha=0.8, linewidth=3, linestyle='dashed', label='sg660',
#                 path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()], zorder=3) 

#         # Plot every 10th glider position
#         for day in [120,125,135, 150, 160, 170, 185, 200]:
#                 i = np.where(dav_660.yearday.round()==day)
#                 ax.scatter(dav_660.lon.loc[i].values, dav_660.lat.loc[i].values, s=100, color='darkorchid', alpha=0.2, zorder=3)
#                 ax.text(dav_660.lon.loc[i].values[0]+.04, dav_660.lat.loc[i].values[0]+0.08, str(day), 
#                                 fontsize=16, color=plat_colors['sg660'],
#                         path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()], zorder=3)


#         # Plot EKE
#         sca = avg_aviso.eke.plot(ax=ax, cmap='bone_r', alpha=0.9, add_colorbar=False, zorder=0, vmin=0.02, vmax=0.58)
#                         # cbar_kwargs={'label': 'EKE ' + r'$\mathregular{[m^2/s^2]}$'})
#         # fig.colorbar(sca, ax=ax).set_label('EKE ' + r'$\mathregular{[m^2/s^2]}$')


#         # Plot ADT Contours
#         contour = ax.contour(avg_aviso['longitude'], avg_aviso['latitude'], avg_aviso['adt'], 
#                         colors='k', alpha=0.3, linestyles='solid', zorder=1, levels=2)
#         ax.clabel(contour, inline=True, fontsize=12)  # Add labels to contour lines

# fig.colorbar(sca, ax=axs[:], shrink=0.4).set_label('EKE ' + r'$\mathregular{[m^2/s^2]}$')

# for ax in axs:
#     ax.yaxis.set_major_formatter("{x:1.0f}°S")
#     ax.xaxis.set_major_formatter("{x:1.0f}°E")
#     ax.set_xlabel('')
#     ax.set_ylabel('')
#     ax.set_aspect('equal')


