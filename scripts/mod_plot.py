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
# import importlib as lib
import seaborn as sns
from importlib import reload

from mod_main import ytd2datetime
# import mod_MLV as mlv
# import mod_DFproc as dfproc
# dav_float = pd.read_csv('../data/bgc-argo/dav_sgfloat_EKEPAR.csv')
# reload(sg)

# %% Plotting parameters
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

# %% Set up units
umol_unit = (r'$\mathbf{[\mu} \mathregular{mol~kg} \mathbf{^{-1}]}$')
# umol_unit = (r'$[\mu \mathregular{mol~kg^{-1}}]$')
eke_unit = (r'$\mathbf{[m^2~s^{-2}]}$')
umol_unit_squared = (r'$\mathbf{[\mu} \mathregular{mol^2~kg} \mathbf{^{-2}]}$')


sigma_unit = (r'$\mathbf{\sigma_0}$ ' + '[kg' + r'$\mathbf{~m^{-3}]}$')
spice_unit = (r'$ \mathbf{\tau} $ ' + '$\mathbf{[m^{-3}~}$' + 'kg]')
backscatter_unit = ('log([m'  + r'$ \mathbf{^{-1}} $' + '])')
par_unit = ('[W m' + r'$ \mathbf{^{-1}} $' + ']')
hb_unit = ('[s' + r'$ \mathregular{^{-2}} $' + ']')
fsle_unit =  ('[days ' + r'$\mathbf{^{-1}}$' + ']')

delta_title = (r'$\mathbf{\Delta }\mathregular{N_{ML}}$ ')
overline_title = (r'$\overline{\mathregular{N}} \mathregular{_{ML}}$ ')
hvar_title = ('s'+ r'$ \mathbf{^2_{H,NO_3}}$ ')
bbp_title = ('bbp' + r'$_{\mathregular{470}} $')
hb_title = ('|' + r'$\mathbf{\nabla_h}\mathregular{b}$' + '|')
# $s^2_{H,NO_3}$
# hvar_title = ('s'+ r'$ $ ')


# Data

# reload(mlv)
# reload(sg)
from mod_main import df_659, df_660, dav_659, dav_660
from mod_main import dav_6030, dav_float, sgfloat
from mod_RFR import RF_validation, RF_test, RF_modelmetrics, RF_featimps
from mod_RFR import RF_kfold, RF_loo


from mod_MLV import mlstat_659, mlstat_660, mlstat_float
from mod_MLV import hvariance_659, hvariance_660


d0_660 = {27.3: 174.62, 27.4: 223.69, 27.5: 296.81, 27.6: 396.96}
isos = [27.3, 27.4, 27.5, 27.6]

# %% RANDOM FOREST TRAINING



# %% RANDOM FOREST VALIDATION



# %% RANDOM FOREST TESTING

def compare_resolution(df_glid = df_660, fsize=(8,6), textsize=16, minday=129, maxday=161, maxpres=750, dateformat=True, vertical=True):
    # Plot Float
    fig, axs = plt.subplots(2,1, figsize=fsize, layout='constrained', sharex=True)

    for ax in axs[0:1]:
        dat = sgfloat[(sgfloat.yearday <maxday) & (sgfloat.yearday > minday)]
        sca = ax.scatter(dat.yearday, dat.pressure, c=dat.nitrate, cmap=cmo.deep, s=80, marker='s' , vmin=24.1, vmax=35.9) #, marker='s')
        ax.set_title('BGC-Argo Float Nitrate', fontsize=textsize)

    for ax in axs[1:]:
        dat = df_glid[(df_glid.yearday <maxday) & (df_glid.yearday > minday)]
        dat = dat[dat.pressure < maxpres]
        sca = ax.scatter(dat.yearday, dat.pressure, c=dat.nitrate_G, cmap=cmo.deep, s=20, vmin=24.1, vmax=35.9) #, vmin=1.5, vmax=3.6) #, marker='s')
    ax.set_title('Glider RFR Nitrate', fontsize=textsize)

    if vertical:
        plt.colorbar(sca, ax=axs[:], shrink=0.5, orientation='vertical').set_label('Nitrate ' + umol_unit, fontsize=textsize)
    else:
        plt.colorbar(sca, ax=axs[:], shrink=0.5, orientation='horizontal').set_label('Nitrate ' + umol_unit, fontsize=textsize)
    ######################
    ### Always ON

    if dateformat:
        list = np.arange(130, 165, 5)
        # list = [ytd2datetime(i) for i in list]
        # list = [np.datetime_as_string(i) for i in list]
        # list = [i[5:10] for i in list]
        # axs[1].set_xticklabels(list)
        axs[1].set_xticks(list)
        axs[1].set_xticklabels(['16-May', '21-May', '26-May', '31-May', 
                                '05-Jun', '10-Jun', '15-Jun'])
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
            sca = axs[0,ind].scatter(df_glid.yearday, df_glid.pressure,c=df_glid.sigma0, cmap=cmo.dense, s=dotsize, zorder=3, 
                vmin=26.9, vmax=27.9)
            # plt.colorbar(sca, ax=axs[0,ind], aspect=13).set_label(sigma_unit, fontsize=font)
        if 'spice' in vlist:
            sca2 = axs[1, ind].scatter(df_glid.yearday, df_glid.pressure, c=df_glid.spice, cmap="Oranges", s=dotsize, zorder=3,
                vmin=-0.65, vmax=0.1)
            # plt.colorbar(sca2, ax=axs[1,ind], aspect=13).set_label(spice_unit, fontsize=font)
        if 'nitrate_G' in vlist:
            sca3 = axs[2, ind].scatter(df_glid.yearday, df_glid.pressure, c=df_glid.nitrate_G, cmap=cmo.deep, s=dotsize, zorder=3, 
                vmin=24.0, vmax=35.8)
            # plt.colorbar(sca3, shrink=0.6, ax=axs[2:4,ind], aspect=16).set_label('Nitrate ' + umol_unit, fontsize=font)
            sca4 = axs[3, ind].scatter(df_glid.yearday, df_glid.sigma0, c=df_glid.nitrate_G, cmap=cmo.deep, s=dotsize, zorder=3, 
                vmin=24.0, vmax=35.8)
    
    plt.colorbar(sca, ax=axs[0,1], aspect=13).set_label(sigma_unit, fontsize=font)
    plt.colorbar(sca2, ax=axs[1,1], aspect=13).set_label(spice_unit, fontsize=font)
    plt.colorbar(sca3, shrink=0.6, ax=axs[2:4,1], aspect=16).set_label('Nitrate ' + umol_unit, fontsize=font)

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
        ax.vlines(150, -20, preslim, color='r', linewidth=1.5, linestyle='dashed', alpha=0.7, zorder=3)
        ax.vlines(170, -20, preslim, color='k', linewidth=1.5, linestyle='dashed', alpha=0.7, zorder=3)
        ax.set_xlim([120, 200])
        ax.invert_yaxis()

    # Set limits
    for ax in axs[0:3,0]:
        ax.set_ylim([preslim+5,-8])
    for ax in axs[3,:]:
        ax.grid(zorder=1, alpha=0.5)

    # Last row limits
    axs[3,0].set_ylabel(sigma_unit)
    axs[3,0].set_ylim([27.87, 26.77])
    axs[3,1].set_ylim([27.99, 26.85])


    # Titles and X labels
    axs[0,0].set_title('SG659', fontsize=font); axs[0,1].set_title('SG660', fontsize=font)
    xt = [120, 140, 160, 180, 200]
    if dateformat: 
        for ax in axs[3,0:2]:
            # ax.set_xticklabels(str(sg.ytd2datetime(k))[-5:] for k in xt)
            ax.set_xticklabels(['1-May', '21-May', '10-Jun', '30-Jun', '20-Jul'])
    else:
        for ax in axs[3,0:2]:
            ax.set_xlabel('[Yearday]')

    plt.show()

    return fig, axs


def plt_time_series(fsize = (8,12), dotsize=10, diasize=60, titlesize=16, dateformat=False, legend=False):
    """ 
    Main time series comparison. """
    fig, axs = plt.subplots(8,1,figsize=fsize, sharex=True, layout='constrained') # 14,12
    axs = axs.flatten()

    ind = 0
    for ax in axs[ind:ind+1]:
        ax.scatter(dav_659.yearday, dav_659.eke_avg, s=dotsize, alpha=0.6, color=plat_colors['sg659'], zorder=3)
        ax.scatter(dav_660.yearday, dav_660.eke_avg, s=dotsize, alpha=0.6, color=plat_colors['sg660'], zorder=3)
        ax.scatter(dav_float.yearday, dav_float.eke_avg,  marker='D', edgecolors='k', linewidth=2.1,
                s=diasize, alpha=0.8, color='#CCBB44', zorder=3)
        ax.set_ylabel(eke_unit) 
        ax.set_title('EKE', fontsize=titlesize)
        # ax.set_ylim([-.05, 0.72]) # for eke
        ax.set_ylim([.01, 0.31]) # for eke_avg
    
    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(mlstat_659.yearday, mlstat_659.mld, s=dotsize, alpha=0.6, color=plat_colors['sg659'], zorder=3)
        ax.scatter(mlstat_660.yearday, mlstat_660.mld, s=dotsize, alpha=0.6, color=plat_colors['sg660'], zorder=3)
        ax.scatter(dav_6030.yearday, dav_6030.mld, label='float',  marker='D', edgecolors='k', linewidth=2.1,
                s=diasize, alpha=0.8, color='#CCBB44', zorder=3)
        
        ax.set_ylabel('[m]')
        ax.invert_yaxis()
        ax.set_ylim([210, -10])
        ax.set_title(('MLD'), fontsize=titlesize)
        
        
    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(mlstat_659.yearday, np.abs(mlstat_659.bx), label='glider 659', s=dotsize, alpha=0.6, color=plat_colors['sg659'], zorder=3)
        ax.scatter(mlstat_660.yearday, np.abs(mlstat_660.bx), label='glider 660', s=dotsize, alpha=0.6, color=plat_colors['sg660'], zorder=3)
        ax.set_ylabel(hb_unit) 
        ax.set_title(hb_title, fontsize=titlesize) # @ (at 100m)
        ax.set_ylim([-5e-8, 0.95e-06])
        # ax.legend(fontsize=16, markerscale=2.5)
    # axs[0].scatter(df_659.yearday, df_659.pressure, c=df_659.nitrate, s=5, cmap=cmo.deep, zorder=1)


    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(mlstat_659.yearday, mlstat_659.nitrate_mean, label='sg659', s=dotsize, alpha=0.6, color=plat_colors['sg659'])
        ax.scatter(mlstat_660.yearday, mlstat_660.nitrate_mean, label='sg660', s=dotsize, alpha=0.6, color=plat_colors['sg660'])
        ax.scatter(mlstat_float.yearday, mlstat_float.nitrate_mean, label='float',  marker='D', edgecolors='k', linewidth=2.1,
                s=diasize, alpha=0.8, color='#CCBB44')
        ax.set_ylabel(umol_unit)
        ax.set_ylim([23.7, 29.5])
        ax.set_title(overline_title, fontsize=titlesize) 

        # to run once for paper fig legend:
        # if legend:
        #     ax.legend(markerscale=1, ncol=3)

    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(mlstat_659.yearday, mlstat_659.nitrate_cross, label='sg659', s=dotsize, alpha=0.6, color=plat_colors['sg659'])
        ax.scatter(mlstat_660.yearday, mlstat_660.nitrate_cross, label='sg660', s=dotsize, alpha=0.6, color=plat_colors['sg660'])
        ax.scatter(mlstat_float.yearday, mlstat_float.nitrate_cross, label='float',  marker='D', edgecolors='k', linewidth=2.1,
                    s=diasize, alpha=0.8, color='#CCBB44')
        ax.set_ylabel(umol_unit)
        ax.hlines(0, 120, 200, linestyle='--', linewidth=2.1, color='k', alpha=0.5)
        ax.set_ylim([-6, 1])
        ax.set_title(delta_title, fontsize=titlesize)
        
    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(hvariance_659.yearday, hvariance_659.nitrate_mean, label='sg659', s=dotsize+4, alpha=0.6, color=plat_colors['sg659'])
        ax.set_ylabel(umol_unit_squared) 
        ax.scatter(hvariance_660.yearday, hvariance_660.nitrate_mean, label='sg660', s=dotsize+4, alpha=0.6, color=plat_colors['sg660'])
        ax.set_ylim([-.05, 0.47])
        ax.set_title(hvar_title, fontsize=titlesize)


    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(mlstat_659.yearday, np.log(mlstat_659.bbp470_mean), label='sg659', s=dotsize, color=plat_colors['sg659'], alpha=0.6)
        ax.scatter(mlstat_660.yearday, np.log(mlstat_660.bbp470_mean), label='sg660', s=dotsize, color=plat_colors['sg660'], alpha=0.6)
        ax.set_ylabel(backscatter_unit) 
        ax.set_ylim([-8.05, -7.2])
        ax.set_title(bbp_title, fontsize=titlesize)

    ind = ind+1
    for ax in axs[ind:ind+1]:
        ax.scatter(dav_659.yearday, dav_659.par, label='par', s=dotsize, alpha=0.6, color=plat_colors['sg659'])
        ax.scatter(dav_660.yearday, dav_660.par, label='par', s=dotsize, alpha=0.6, color=plat_colors['sg660'])
        ax.scatter(dav_float.yearday, dav_float.par, label='par', s=diasize, alpha=0.8, color=plat_colors['float'],
                marker='D', edgecolors='k', linewidth=2.1)
        ax.set_ylabel(par_unit) # PAR
        ax.set_ylim([0.6, 9.1])
        ax.set_title('PAR', fontsize=titlesize) 


    xt = np.arange(120,210,10)
    if dateformat: 
        for ax in axs[-1:]:
            ax.set_xticklabels(str(ytd2datetime(k))[-5:] for k in xt)
            ax.set_xticklabels(['01-May', '11-May', '21-May', '31-May',
                                '10-Jun', '20-Jun', '30-Jun', 
                                '10-Jul', '20-Jul'])
    else:
        for ax in axs[-1:]:
            ax.set_xlabel('[Yearday]')

    for ax in axs:
        ax.grid(zorder=0, alpha=0.4)
        ax.set_xlim([120,200])
        # ax.legend()

        ax.vlines(150, -30,250, color='r', linewidth=1.8, alpha=0.5, linestyle='dashed', zorder=1)
        ax.vlines(170, -30,250, color='k', linewidth=1.8, alpha=0.5, linestyle='dashed', zorder=1)
    
    return fig, axs


# %% WAVELET

def plot_cwt(sig, frequency, xtime, coi, zamp, signif_amp, ax, cmap = cmo.amp, dateformat=False, ekeline=True):

    contourf_args = {'cmap':cmap, 'vmax': 1.3, 'origin': 'lower', 'levels': 11}
    # make cwt
    # ax = axs[ind,0]
    y_axis = 1/frequency
    ylabel = 'Period [days]'

    yticks_default = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])
    mask = (yticks_default >= np.min(y_axis)) & (yticks_default <= np.max(y_axis))
    yticks = yticks_default[mask]

    cont = ax.contourf(xtime,y_axis, zamp, **contourf_args)
    ax.set_yscale('log')
    cbar_args = {'drawedges': False, 'orientation': 'vertical', 'fraction': 0.15, 'pad': 0.05}
    cbar_style={'label':'Amplitude', 'ticks':[0.0, 0.3, 0.6, 0.9, 1.2]}
    cbar_args.update(cbar_style)
    cb = plt.colorbar(cont, **cbar_args)

    # plot coi 
    ax.plot(xtime, coi, 'k--')
    ax.fill_between(xtime, coi, np.max(coi), color='white', alpha=0.6)

    signif_clr='darkslategray'; signif_linestyles='-'; signif_linewidths=1
    signif_method_label = {'ar1': 'AR(1)'}
    signif_boundary = zamp / signif_amp
    ax.contour(
        xtime, y_axis, signif_boundary, [-99, 1],
        colors=signif_clr,
        linestyles=signif_linestyles,
        linewidths=signif_linewidths,
    )
    ax.set_title(r'$\mathbf{\sigma_0}=$' + str(sig) + ', ' + r'$\mathregular{\bar{d}}=$' + str(int(d0_660[sig])) + ' m',
            fontsize=16)
    ylim = [np.min(y_axis), np.min([np.max(y_axis), np.max(coi)])]
    ax.set_ylim(ylim)

    if ekeline:
        ax.vlines(150, 0,50, color='r', linewidth=1.5, linestyle='dashed', alpha=0.65, zorder=3)
        ax.vlines(170, 0,50, color='k', linewidth=1.5, linestyle='dashed', alpha=0.65, zorder=3)

    ax.set_yticks(yticks)
    ax.set_yticklabels(['0.2', '0.5', '1', '2', '5', '10', '20'])

    xt = [120, 140, 160, 180, 200]
    if dateformat: 
        ax.set_xticks(xt)
        ax.set_xticklabels(['1-May', '21-May', '10-Jun', '30-Jun', '20-Jul'])
    else: 
        ax.set_ylabel('Period [days]')

    return ax