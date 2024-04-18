# Random Forest Regression on multi-platform, in-situ ocean observations

Paper in submission to AMS Artifical Intelligence in Earth Sciences (AIES) journal, pending review. [pre-print manuscript available by request]


Contact: Song Sangmin <sangsong@uw.edu> <github: song-sangmin>

University of Washington, School of Oceanography

Last updated: Apr 16 2023

## Overview


### Motivation / Research Abstract 

Nutrient cycling in the ocean is mediated by physical mixing processes that span diverse spatial and temporal scales. New biogeochemical profiling floats (BGC-Argo) have begun to collect nutrient data on a global scale, but their 10-day cycling period limits what processes they capture. Small-scale dynamics, occurring on $\mathcal{O}$(1) day and $\mathcal{O}$(1) km, remain particularly difficult to observe in-situ. Here, we demonstrate that a Random Forest machine learning approach can recover high-frequency information by bridging the sampling strategies of different ocean profilers. Our regressor is trained and tested on BGC-Argo data to within approx. 3% accuracy, then applied to observations from two rapid-sampling Seagliders deployed during the Southern Ocean Glider Observations of the Submesoscale (SOGOS) experiment in 2019. This approach generates novel nitrate distributions at 50 times the horizontal resolution of original float data. Using the high-resolution outputs, we then identify signatures of biogeochemical tracer injection that coincide with enhanced stirring in a turbulent region downstream of the Southwest Indian Ridge. These intermittent transport events occur during a period of increased nutrient drawdown and primary production. By synthesizing information from multiple platforms, Random Forest extends the capabilities of the global BGC-Argo array and allows for deeper understanding of biogeochemical cycling across scales. We present Random Forest regression as a powerful and accessible tool that can be generalized to suit various observing systems.



### Notebooks

Three main notebooks describe the (1) Random Forest model development, (2) application to ocean data and performance evaluation, followed by (3) scientific analysis of the output. 

These should ideally be run in the following order:


1. Random Forest Training: `scripts/Training_RandomForest.ipynb`

- This notebook demonstrates our machine learning approach to ocean nutrient prediction. We explore different variations of a Random Forest regressor using data from autonomous ocean profilers [see AIES paper for more detail]. 

- The process of model development includes three main phases: training, validation, and testing. The models are trained on different feature lists, then validated to screen for the best model parameters. We stress that additional cross-validation metrics are needed for observations with spatiotemporal biases (Stock et al. 2022).

- The testing phase uses withheld float data from the SOGOS BGC-Argo float to estimate the model error. We also compare performance with two well-established machine learning models, ESPER-Mixed and CANYON-B, which both rely on neural networks. 



2. Random Forest Analysis: `scripts/Analysis_RandomForest.ipynb`

- Here, we

- 


## Code Directory


### Scripts

Some major functions are stored in modules, which are used to control processing functions that may be needed across scripts. 

- *Modules*: `sgmod_[purpose].py`: 


                sgmod_main      as sg             Common glider functions
                sgmod_L3proc    as gproc          Used for xarray Datset processing of the level 3, 'L3' gridded glider product
                sgmod_DFproc    as dfproc         Used for pandas Dataframe processing during analysis
                sgmod_plotting  as sgplot         Used to define common plotting parameters


- *Classes*:
                class_RF        as crf            Object for Random Forest training
                

### Folders


- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `figures/` : diagnostic and analysis figures


### Naming Conventions

 
- *Glider:*    
                dav_659                         Profiles-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                profid                          Unique name for each profile

- *Floats:*    
                dav_659                         Profile-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                wmoid                           Unique name for each float WMO#
                profid                          Unique name for each profile



### Data Sources

<!-- 
- [Link to gridded variables](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/Et5YKAWyry5KkSst28_unxsBE3Vc5TCbOGl-3lR4sTvSQQ?email=joycecai%40uw.edu&e=einIE4)

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
- Description of Argo float and ship are in first paper from SOGOS program: [Link to Dove et. al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JC017178)



### Other Notes

Two phases of processing were developed externally in MATLAB: 

1. Oxygen optode time response correction (Adapted from Yui Takeshita, MBARI)
2. ESPER (Courtesy of Brendan Carter [Link to Github])


ACC Front locations courtesy of Jade Sauve. 