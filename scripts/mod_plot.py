
import xarray as xr
import pandas as pd
import numpy as np
import gsw 
from cmocean import cm as cmo
from datetime import datetime
import scipy

from importlib import reload

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patch
from labellines import labelLine, labelLines
import matplotlib.colors as mpcolors
from matplotlib.ticker import FormatStrFormatter


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

model_list = ['Model_A', 'Model_B', 'Model_C', 'Model_D', 'Model_E', 'Model_F', 'Model_G']
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


# %% Data
from mod_main import df_659, df_660, dav_659, dav_660
from mod_main import dav_6030, dav_float, sgfloat
from mod_main import altimetry

from mod_main import shipDF, floatDF
wmoids = floatDF.wmoid.unique()

from mod_RFR import RF_validation, RF_test, RF_modelmetrics, RF_featimps
from mod_RFR import RF_kfold, RF_loo


from mod_MLV import mlstat_659, mlstat_660, mlstat_float
from mod_MLV import hvariance_659, hvariance_660

d0_660 = {27.3: 174.62, 27.4: 223.69, 27.5: 296.81, 27.6: 396.96}
isos = [27.3, 27.4, 27.5, 27.6]


# %% STUDY REGION

def study_region_eke(df_float = sgfloat, df_glid1  = df_659, df_glid2 = df_660, fs = (7,8), fontsize=14):
    """ 
    @param:         fontsize    label font size
    """

    fig, axs = plt.subplots(2,1, figsize=fs, layout='constrained')
    axs = axs.flatten()
    df_float = df_float[df_float.yearday<205]

    # START HERE
    for ax in [axs[0]]:
            datestart='2019-04-30'
            dateend='2019-07-25'
            lat1=-56.8; lat2=-43
            lon1=19; lon2=41
            data_sat = altimetry.sel(time=slice(datestart, dateend))
            data_sat = data_sat.sel(latitude=slice(lat1, lat2))
            data_sat = data_sat.sel(longitude=slice(lon1, lon2))
            avg_altimetry = data_sat.mean(dim='time')

            #background EKE
            sca = avg_altimetry.eke.plot(ax=ax, cmap='bone_r', alpha=0.8, add_colorbar=False, vmin=0.02, vmax=0.29)
            # fig.colorbar(sca, ax=ax, shrink=0.5).set_label('EKE')

            contour = ax.contour(avg_altimetry['longitude'], avg_altimetry['latitude'], avg_altimetry['adt'], 
                                    colors='k', alpha=0.3, linestyles='solid', zorder=1, levels=4)
            ax.clabel(contour, inline=True, fontsize=fontsize-2)  # Add labels to contour lines

            # Plot float

            ax.plot(df_float.lon,df_float.lat, color =  '#CCBB44', alpha=1, linewidth=4, linestyle='dashed',
                    label='float', zorder=3)  # sns.color_palette("Oranges")[3]
            ax.scatter(df_float.lon,df_float.lat, color =  '#CCBB44', alpha=0.9, s=50, 
                    marker='D', edgecolors='k', zorder=3)

            # Gliders
            ax.plot(df_glid1.lon, df_glid1.lat, color = plat_colors['sg659'], alpha=0.8, linewidth=6, linestyle='dashed', label='sg659', 
                    path_effects=[pe.Stroke(linewidth=8, foreground='k'), pe.Normal()], zorder=3) 

            ax.plot(df_glid2.lon, df_glid2.lat, color = plat_colors['sg660'], alpha=0.8, linewidth=4, linestyle='dashed', label='sg660', 
                    path_effects=[pe.Stroke(linewidth=6, foreground='w'), pe.Normal()], zorder=3) 
            
            # Box around study region in panel b
            rect = patch.Rectangle((29.5,-54.7), 9.9, 5.5, fill=False, color = 'k', alpha=0.3,linewidth=2, zorder=1)
            ax.add_patch(rect)

    for ax in [axs[1]]:
            datestart='2019-04-30'
            dateend='2019-07-25'
            lat1=-54.7; lat2=-49.2
            lon1=29.5; lon2=39.4
            data_sat = altimetry.sel(time=slice(datestart, dateend))
            data_sat = data_sat.sel(latitude=slice(lat1, lat2))
            data_sat = data_sat.sel(longitude=slice(lon1, lon2))
            avg_altimetry = data_sat.mean(dim='time')

            # Plot float
            ax.plot(dav_float.lon,dav_float.lat, color =  '#CCBB44', alpha=1, linewidth=4, linestyle='dashed',
            label='float', zorder=3)  # sns.color_palette("Oranges")[3]
            ax.scatter(dav_float.lon.values,dav_float.lat.values, color =  'k', alpha=0.9, s=100, 
                    marker='D', edgecolors='k', zorder=3)
            ax.scatter(dav_float.lon.values,dav_float.lat.values, color =  '#CCBB44', alpha=0.9, s=70, 
                    marker='D', edgecolors='k', zorder=3)
            
            for ind, day in enumerate([int(x) for x in dav_float.yearday.values]):
                    if day in ([120,135,140,150, 160, 170, 185, 205]):
                            ax.text(dav_float.lon.loc[ind]-.3, dav_float.lat.loc[ind]-.4, str(day), 
                                            fontsize=fontsize, color='k',
                            path_effects=[pe.Stroke(linewidth=3, foreground='#CCBB44'), pe.Normal()], zorder=3)

            # Plot Glider
            ax.plot(df_glid2.lon, df_glid2.lat, color = plat_colors['sg660'], alpha=0.8, linewidth=3, linestyle='dashed', label='sg660',
                    path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()], zorder=3) 

            # Plot every 10th glider position
            for day in [120,125,135, 150, 160, 170, 185, 200]:
                    i = np.where(dav_660.yearday.round()==day)
                    ax.scatter(dav_660.lon.loc[i].values, dav_660.lat.loc[i].values, s=100, color='rebeccapurple', alpha=0.2, zorder=3)
                    ax.text(dav_660.lon.loc[i].values[0]+.04, dav_660.lat.loc[i].values[0]+0.08, str(day), 
                                    fontsize=fontsize, color=plat_colors['sg660'],
                            path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()], zorder=3)


            # Plot EKE
            sca = avg_altimetry.eke.plot(ax=ax, cmap='bone_r', alpha=0.9, add_colorbar=False, zorder=0, vmin=0.02, vmax=0.29)
                            # cbar_kwargs={'label': 'EKE ' + r'$\mathregular{[m^2/s^2]}$'})
            # fig.colorbar(sca, ax=ax).set_label('EKE ' + r'$\mathregular{[m^2/s^2]}$')


            # Plot ADT Contours
            contour = ax.contour(avg_altimetry['longitude'], avg_altimetry['latitude'], avg_altimetry['adt'], 
                            colors='k', alpha=0.3, linestyles='solid', zorder=1, levels=2)
            ax.clabel(contour, inline=True, fontsize=fontsize-2)  # Add labels to contour lines

    fig.colorbar(sca, ax=axs[:], shrink=0.4).set_label('EKE ' + r'$\mathbf{[m^2~s^{-2}]}$')

    for ax in axs:
        ax.yaxis.set_major_formatter("{x:1.0f}°S")
        ax.xaxis.set_major_formatter("{x:1.0f}°E")
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect('equal')
    return axs[0]

# %% ARGO MAP
# Download fronts of the Antarctic Circumpolar Current

def training_float_map(floatDF = floatDF, shipDF = shipDF, ax=None, fsize = (10,8), fontsize=16):
    """
    """
    PF = pd.read_csv('../data/ACC_fronts/PF.csv', header=None)
    SIF = pd.read_csv('../data/ACC_fronts/SIF.csv', header=None)
    for csv in [PF, SIF]:
        csv.columns = ['lon', 'lat']

    if ax == None:
        fig = plt.figure(figsize=fsize)
        ax=plt.gca()

    for wmo in wmoids[wmoids!=5906030]:
        ax.plot(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.2, linewidth=3, label=str(wmo)[3:], zorder=3)
        ax.scatter(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = wmo_colors[wmo], alpha=0.1, s=20, zorder=3)

    for wmo in [5906030]:
        ax.plot(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = sns.color_palette("Oranges")[3], alpha=0.4, linewidth=5, label=str(wmo)[3:], zorder=3)
        ax.scatter(floatDF[floatDF.wmoid==wmo].lon,floatDF[floatDF.wmoid==wmo].lat,
                color = sns.color_palette("Oranges")[3], alpha=0.1, s=20, zorder=3)

    for front in [SIF, PF]:
        front = front[(front.lon > 0 ) & (front.lon <65)]
        ax.scatter(front.lon, front.lat, color='k', alpha=0.2, s=4, zorder=3)
    #      ax.plot(front.lon, front.lat, color='k', alpha=0.1, linewidth=2, zorder=3)

    # plt.title('BGC-Argo Training Data')
    # plt.title('Argo Yeardays: ' + str(start_yd) + ' to ' + str(end_yd))
    rect = patch.Rectangle((29,-55),10,5, fill=True, color="orange", alpha=0.3,linewidth=2, zorder=1)

    ax.add_patch(rect)
    ax.set_xlim(1,69)
    ax.set_ylim(-62,-47)
    ax.set_aspect('equal')

    ax.yaxis.set_major_formatter("{x:1.0f}°S")
    ax.xaxis.set_major_formatter("{x:1.0f}°E")

    ax.grid(linestyle='dashed', alpha=0.6, zorder=1)

    lines = ax.get_lines()
    labelLines(lines, align=False, fontsize=fontsize, zorder=3)
    # labelLines(lins) # , align=False, fontsize=fontsize)
    ax.plot(shipDF.lon, shipDF.lat, alpha=0.8, linestyle='dashed', c='k', linewidth=3)
    return ax 


def argo_time_coverage(floatDF = floatDF, fsize=(10,5), fontsize=14, ax=None):
    sns.set_palette(wmo_colors.values())

    if ax == None:
        fig = plt.figure(figsize=fsize)
        ax = plt.gca()
    
    pd.DataFrame({k: v for k, v in floatDF.groupby('wmoid').yearday}).plot.hist(stacked=True, ax=ax, zorder=3, alpha=0.8, linewidth=1, edgecolor='k') 
    ax.legend([str(wmo)[-4:] for wmo in wmoids], loc='upper left', fontsize=fontsize)

    # ax.set_title('Time Coverage by Float WMO')
    # ax.set_xlabel('Time')
    ax.set_xticks([-365, 0, 365, 365*2])
    ax.set_xticklabels(['Jan 2018', 'Jan 2019', 'Jan 2020', 'Jan 2021'])
    ax.set_ylim([0, 3000])
    ax.set_ylabel('Observations')
    ax.grid(axis='y', alpha=0.5, zorder=0)

    return ax

# %% BASIC SECTION PLOTS

def time_pres_section(data, var, figsize = (10,5), ax = None, cmap='RdBu_r', vlims = None, xlims=None, ylims=None):
    if ax == None:
        fig  = plt.figure(figsize=(12,6), tight_layout=True)
        ax = plt.gca()

    if vlims == None:
        sca = ax.scatter(data.yearday, data.pressure, c=data[var], cmap=cmap, s=250, marker='s')
    else:
        sca = ax.scatter(data.yearday, data.pressure, c=data[var], cmap=cmap, s=250, marker='s', vmin=vlims[0], vmax=vlims[1])

    ax.invert_yaxis()
    ax.set_ylabel('Pressure')
    ax.set_xlabel('Yearday')

    if xlims != None:
        ax.set_xlim(xlims)
    if ylims != None:
        ax.set_ylim(ylims)
    
    return ax



# %% RANDOM FOREST TRAINING



# %% RANDOM FOREST VALIDATION

def kfold_KDE(data, model_list = model_list, textsize=14):
    """
    New KDE plot using all combined validation errors from K-Fold 
    @param:     data        RF_validation output (cv_kfold.val_error)

    """ 
    tag = 'Predicted - Observed Nitrate'
    var = 'val_error'; ymax=2.12
    # ymax = 2.1


    # Make Figure with 2 Subplots
    fig, axs = plt.subplots(1, 2, figsize=(11,4), layout='constrained')
    axs=axs.flatten()

    for ax in axs[0:1]:
        for mod in model_list[:]: 
            RF = data[mod].values

            # Add Gaussian KDE to estimate probability density function
            x = np.linspace(RF.min(), RF.max(), 1000)
            kde = scipy.stats.gaussian_kde(RF)

            lw = 2; ls = 'solid'
            if mod == 'Model_G':
                lw=lw+1
            ax.plot(x, kde(x), color=model_palettes[mod], linewidth=lw, linestyle=ls, label=mod[-1], alpha=0.6)

        if var == 'val_error':
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([0, ymax])

        leg = ax.legend(fontsize=14, framealpha=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.5)
        

    for ax in axs[1:]:
        for mod in model_list[3:]: # ['Model_A', 'Model_G']: #
            RF = data[mod].values

            # Gaussian KDE
            x = np.linspace(RF.min(), RF.max(), 1000)
            kde = scipy.stats.gaussian_kde(RF)

            lw = 2; ls = 'solid'
            if mod == 'Model_G':
                lw=lw+1
            if mod in ['Model_D', 'Model_F']:
                ls = '--'
            plt.plot(x, kde(x), color=model_palettes[mod], linewidth=lw, linestyle = ls, label=mod[-1], alpha=0.6)

        ax.set_xlim([-.19, .19])
        ax.set_ylim([1.48, ymax])

        # Legend
        leg = plt.legend(fontsize=textsize, framealpha=1)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.5)

    umol_unit = (r'$\mathbf{[\mu} \mathregular{mol~kg} \mathbf{^{-1}]}$')


    for ax in axs:
        # Show float uncertainty and 0 line
        ax.axvline(x=0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
        ax.axvline(x=-0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
        ax.vlines(0, ymin=0, ymax=3, colors='k', alpha=0.8, linewidth=2, linestyle='dotted', zorder=1)
        # ax.set_ylim([0,ymax])
        ax.grid(alpha=0.55, zorder=1)

        ax.set_ylabel('Density')
        ax.set_xlabel("Nitrate Error " + umol_unit)
        ax.set_title('K-Fold Validation Errors', fontsize=textsize)

    return ax


# def kfold_bplot(data, figsize=(8,4), fontsize=14):
    """ 
    Single version (left panel)
    """
#     fig, ax = plt.subplots(figsize=(8,4))

#     lw= 1.5
#     bplot = ax.boxplot(data.values(), widths=0.65, vert=False, patch_artist=True,
#                     medianprops = {'color':'k', 'linewidth':lw},
#                     capprops= {'color':'k', 'linewidth':lw},
#                     flierprops= {'color':'k', 'linewidth':lw},
#                     boxprops = {'color':'k', 'linewidth':lw})
#     ax.set_yticklabels([x[-1] for x in data.keys()])
#     ax.grid(zorder=1, alpha=0.5)
#     ax.invert_yaxis()
#     ax.set_ylabel('Model')
#     ax.set_xlabel('Fold Validation MAE [umol/kg]')

#     colors = []
#     for mdl in model_list:
#         colors.append(model_palettes[mdl])

#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.5))

#     plt.show()
#     return ax 

def kfold_boxplots(data, bias_one, bias_two, figsize=(12, 6), modlist = ['Model D', 'Model_G'], textsize=14):
    # Make Figure with 2 Subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize, layout='constrained', width_ratios=[1.5,1])
    axs=axs.flatten()

    for ax in axs[:1]:
        
        lw= 1.5
        bplot = ax.boxplot(data.values(), widths=0.65, vert=False, patch_artist=True,
                        medianprops = {'color':'k', 'linewidth':lw},
                        capprops= {'color':'k', 'linewidth':lw},
                        flierprops= {'color':'k', 'linewidth':lw},
                        boxprops = {'color':'k', 'linewidth':lw})
        ax.set_yticklabels([x[-1] for x in data.keys()])
        ax.grid(zorder=1, alpha=0.5)
        ax.invert_yaxis()
        ax.set_ylabel('Model', fontsize=16)
        ax.set_xlabel("Nitrate Error " + umol_unit)

        colors = []
        for mdl in model_list:
            colors.append(model_palettes[mdl])

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.5))
        # ax.axvline(x=0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)

    for ax in axs[1:]:
        color_one = model_palettes[modlist[0]] # sns.color_palette("Purples")[3]
        color_two = model_palettes[modlist[1]]


        bplot_one = ax.boxplot(bias_one.values(), vert=False, showfliers=False, widths=0.6, 
                            patch_artist=True, 
                            medianprops= {'color':color_one, 'linewidth':2},
                            capprops={'color':color_one, 'linewidth':1.5},
                            whiskerprops={'color':color_one, 'linewidth':1.5},
                            flierprops={'color':color_one, 'linewidth':1.5},
                            boxprops = {'color':color_one, 'linewidth':1},
                            zorder=2)
        for patch, color in zip(bplot_one['boxes'], [color_one]*10):
            patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))


        bplot_two = ax.boxplot(bias_two.values(), vert=False, showfliers=False, widths=0.35, 
                            patch_artist=True, 
                            medianprops= {'color':'k', 'linewidth':2},
                            capprops={'color':color_two, 'linewidth':2},
                            whiskerprops={'color':color_two, 'linewidth':2},
                            flierprops={'color':color_two, 'linewidth':2},
                            boxprops = {'color':'k', 'linewidth':1},
                            zorder=3)
        for patch, color in zip(bplot_two['boxes'], [color_two]*10):
            patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.4))

        ax.invert_yaxis()
        ax.axvline(x=0, color='k', linestyle='dotted', linewidth=3, alpha=0.4, zorder=0)
        

        labels = bias_one.keys()
        ax.set_yticks(range(1, len(labels) + 1), labels, fontsize=textsize)
        ax.set_ylabel("Depths [m]", fontsize=textsize)
        ax.set_xlabel("Nitrate Error " + umol_unit, fontsize=textsize)
        ax.set_xlim([-.99, .99])

        # ax.set_title("Cross-Validation Errors", fontsize=textsize)

        # ax.legend([bplot_one["boxes"][0], bplot_two["boxes"][0]], [mod_one[-1], mod_two[-1]], loc='lower right', fontsize=15)
        ax.grid(axis='x', zorder=1, alpha=0.4)

        ax.axvline(x=0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
        ax.axvline(x=-0.5, color='r', linestyle='dotted', linewidth=2, alpha=0.6, zorder=0)
    return ax

def loo_boxplots(data, figsize=(8,4), textsize=14, lw=1.5):
    """ 
    @param  data:    RF_loo output (cv_kfold.MAEs
            lw:     linewidth of boxplot
    )"""
    fig, ax = plt.subplots(figsize=figsize)

    bplot = ax.boxplot(data.values(), widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'linewidth':lw},
                    boxprops = {'color':'k', 'linewidth':lw})
    ax.set_yticklabels([x[-1] for x in data.keys()])
    ax.grid(zorder=1, alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylabel('Model')
    ax.set_xlabel('Fold Validation MAE [umol/kg]')

    colors = []
    for mdl in model_list:
        colors.append(model_palettes[mdl])

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.5))

    return ax

def loo_boxplots_wmodot(data, data_metrics, modlist = ['Model_D', 'Model_G'], figsize=(8,4), textsize=14, lw=1.5):
    """ 
    @param  data:               RF_loo output (cvloo)
            data_metrics:       RF_modelmetrics output (cvloo_metrics)
            lw:                 linewidth of boxplot
    )"""
    data = {k: data[k] for k in modlist}
    fig, ax = plt.subplots(figsize=(7,4))

    lw= 1.5
    bplot = ax.boxplot(data.values(), widths=0.65, vert=False, patch_artist=True,
                    medianprops = {'color':'k', 'linewidth':lw},
                    capprops= {'color':'k', 'linewidth':lw},
                    flierprops= {'color':'k', 'linewidth':lw},
                    boxprops = {'color':'k', 'linewidth':lw})
    ax.set_yticklabels([x[-1] for x in data.keys()])

    ax.grid(axis='x', zorder=1, alpha=0.5)
    ax.set_ylabel('Model')
    ax.set_xlabel('Spatial LOO MAE ' + umol_unit)
    ax.set_xticks([.2, .3, .4, .5, .6, .7])


    mod_one = modlist[0]
    mod_two = modlist[1]

    colors = [model_palettes[mod_one], model_palettes[mod_two]]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(mpcolors.to_rgba(color, alpha=0.2))

    wmos = [x for x in wmoids if x != 5906030] 
    for wmo in wmos: 
            ax.scatter(data_metrics.at[wmo,mod_one],1, color=wmo_colors[wmo], s = 40, label= str(wmo)[-4:], zorder=3)
            ax.scatter(data_metrics.at[wmo,mod_two],2, color=wmo_colors[wmo], s=40, zorder=3)

    ax.legend(loc='upper right', fontsize=14)
    ax.set_xlim([.17, .85])
    ax.invert_yaxis()

    return ax

        
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
            ax.set_xticks(xt)
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