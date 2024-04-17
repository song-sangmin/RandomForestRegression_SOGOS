# Random Forest Regression on multi-platform, in-situ ocean observations

Paper in submission to AMS Artifical Intelligence in Earth Sciences (AIES) journal, pending review. [pre-print manuscript available by request]

Contact: Song Sangmin <sangsong@uw.edu> <github: song-sangmin>
University of Washington, School of Oceanography
Last updated: Apr 16 2023

## Overview

Three main notebooks describe the (1) Random Forest model development, (2) application to ocean data and performance evaluation, followed by (3) scientific analysis of the output. 

1. Random Forest Training: `scripts/Training_RandomForest.ipynb`

- This notebook demonstrates our machine learning approach to ocean nutrient prediction. We explore different variations of a Random Forest regressor using data from autonomous ocean profilers [see AIES paper for more detail]. The models are first trained on different feature lists, then used for nutrient prediction along the glider track.

- The process of model development includes three main phases: training, validation, and testing. We stress that additional cross-validation metrics are needed for observations with spatiotemporal biases (Stock et al. 2022). 

2. Random Forest Analysis: `scripts/Analysis_RandomForest.ipynb`

- Here, we

- 

**Modules**

Many major functions are stored in modules, which are listed under the Code Directory.

- `sgmod_[purpose].py`: Modules are used to control processing functions required across multiple scripts.
                sgmod_main      as sg             Common glider functions
                sgmod_L3proc    as gproc          Used for xarray Datset processing of the level 3, 'L3' gridded glider product
                sgmod_DFproc    as dfproc         Used for pandas Dataframe processing during analysis
                sgmod_plotting  as sgplot         Used to define common plotting parameters

Two phases of processing were developed externally in MATLAB: 

1. Oxygen optode time response correction (Adapted from Yui Takeshita, MBARI)
2. ESPER (Courtesy of Brendan Carter [Link to Github])

Results of processing are provided in the folder `data/glider`.
---

**Folders**

- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `figures/` : diagnostic and analysis figures

<!-- 

Note* 
- To get to "standard" dec 23 version: (7 floats)
                        
                        Pull in larger argo file from float_processing.ipy

---
**Variable Naming Conventions**
    
*Satellite:*    
 
                fsle_backwards                  map of daily FSLE snapshots
                satellite_data                  adt, geo velocities, ssh anomalies 
*Glider:*    
 
                gp_659                          L3 dataset gridded on pressure
                gi_659                          L3 dataset gridded on isopycnals
                dav_659                         L3 dive-averaged metrics, incl. MLD
                df_659                          Flattened dataframe

                profid                          Unique name for each profile



 ---
**Data Sources**

<!-- [Link to gridded variables](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/Et5YKAWyry5KkSst28_unxsBE3Vc5TCbOGl-3lR4sTvSQQ?email=joycecai%40uw.edu&e=einIE4)

        - `gp_659_forMLGeo1026.nc`  (pressure-gridded 1m, glider #659)
        - `gp_660_forMLGeo1026.nc`  (pressure-gridded 1m, glider #660)

        - `gi_659_forMLGeo1026.nc`  (isopycnal-gridded .001, glider #659)
        - `gi_659_forMLGeo1026.nc`  (isopycnal-gridded .001, glider #659)

                - 'fsle_backwards.nc'           (1-day FSLE from AVISO)
                - 'satellite_data.nc'           (ADT product from AVISO)
 -->
- [Link to float data](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/Es-ESkVfIlpHhpFq7o5LTaoBtqv6pWj6rntxMyXieLEq8A?e=FeRRjs)



- [Argo ERDDAP Data Server](http://www.argodatamgt.org/Access-to-data/ERDDAP-data-server)



- [Link to ship data](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/ErLtPwS6pdZClgo0Flp9lq8Bz73FRmUlhR2zf329gDH-3w?e=hCzidh)
- [GO-SHIP I6 Cruise Line Bottle Data](https://cchdo.ucsd.edu/cruise/325020190403)
- Description of glider data variables are in `Seaglider_DataGuide.pdf`
- Description of Argo float and ship are in first paper from SOGOS program: [Link to Dove (2021)]https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JC017178)


 ---
 ## SOGOS_Apr19 L2 and L3 reprocessing notes

- owner: gbs2@uw.edu - 2021/09/24
- emailed to me (sangsong@uw.edu 2022/10/21)

## SG659
For SG659, glider .eng files for dives 27 through the end were reconstructed from scicon files for
depth and compass (see create_eng.py for details).  The following files:

sc0078b/psc6590078b_depth_depth.eng 
sc0096b/psc6590096b_depth_depth.eng 
sc0176b/psc6590176b_depth_depth.eng 
sc0332b/psc6590332b_depth_depth.eng 
sc0451b/psc6590451b_depth_depth.eng 
sc0476a/psc6590476a_depth_depth.eng 
sc0485a/psc6590485a_depth_depth.eng 

did not exist, so .eng files for those dives were not created.

## SG659 and SG660
Both glider data was the processed the IOP in-house version of the basestation
code.  This version is substantially the same as version 2.13 - released to
Hydroid in May of this year, but not yet available (that I know of).  The
primary change is the incorporation of the FMS system
(https://digital.lib.washington.edu/researchworks/handle/1773/44948) which
yields a number of improvements in the CTD corrections. 

The Hydroid version of the Seaglider flight code and basestation have some
differences in netcdf variable naming.  To process the optode and wetlabs data,
is renamed these variable columns - see the sg_calib_constants.m in both
gliders directories for the remapping scheme.

Unfortunately, our version of the basestation does not apply the PAR sensor
calibrations to the output, so the PAR data is as reported by the instrument.

## L23 Processing
To have a sanity check of the processing, I ran the re-processed data through
code to create L2 (1m binned) and L3 (interpolated with outliers removed) data
sets - see SOGO_Apr19_L23 for the data, and SOGO_Apr19_L23/plots for the output
plots. 

In both gliders data sets, something changed dramatically at the end of the
mission - for SG659 starting on dive 463 and for SG660 on dive 510, the
temperature jumps dramatically.  I didn't trace this all the way back to the
raw instrument output, but observed it is in the original netcdfs provided with
SG660, so I'm assuming this is a real and not processing related.


# Oxygen processing
See 2022-09 Lab Notes
