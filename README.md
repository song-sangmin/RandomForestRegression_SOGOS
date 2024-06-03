# Random forest regression on multi-platform in-situ ocean observations

Paper submitted to to AMS Artifical Intelligence in Earth Sciences (AIES) journal May 2024, pending review. - [Link to pre-print](https://doi.org/10.22541/essoar.171707849.91867565/v1)

Contact: Song Sangmin <sangsong@uw.edu>

University of Washington, School of Oceanography

Last updated: May 30 2024

## Overview

Here, we use a regional random forest regression (RFR) to leverage data from multiple ocean observing instruments that offer different advantages. In our study of the Southern Ocean, we use RFR to produce new nutrient maps at 50 times higher resolution than previously possible. By estimating small-scale information, RFR reveals interactions between physical and biological processes during rapid mixing events that are normally difficult to observe. These short-lived interactions appear to be important in determining local nutrient content and therefore biological activity in this important ocean basin. 


Specifically, our work will go through these main steps: 

1) Preparing BGC-Argo data for our model training data.
2) Training the model on available BGC-Argo and shipboard data. 
3) Validating different models to select optimal parameters; Cross-validating to minimize overfitting.
4) Testing the RFR for an estimation of error. 
5) Applying Seaglider inputs to predict new nitrate distributions. 


Figure: Increasing the resolution of Southern Ocean nitrate maps with RFR.
![Figure](./images/small_resolution.png))

Given increasing observational coverage of the global oceans by Argo floats and other drifting profilers, RFR presents opportunities to derive additional value from these sometimes incomplete biogeochemical datasets. Such efforts to bridge observational gaps using new ocean technologies and machine learning techniques will expand our knowledge of global biogeochemical cycles at previously inaccessible scales.

### Abstract 

Nutrient cycling in the ocean is mediated by physical mixing processes that span diverse spatial and temporal scales. Examining the full range of these transport processes will be critical for understanding controls on oceanic primary production, and therefore carbon and climate systems more broadly. Although new biogeochemical profiling floats (BGC-Argo) have begun to observe nutrient distributions globally, their 10-day cycling period limits the types of physical processes they can capture. Small-scale dynamics (occurring on $\mathcal{O}$(1) day and $\mathcal{O}$(1) km) remain particularly difficult to observe in-situ. Since theory and simulation support that these small-scale motions play significant roles in nutrient transport (Levy et al. 2018, Mahadevan 2016), new data science approaches are needed to assess their impact using available observations. Here, we show that random forest regression (RFR) is an effective machine learning tool for recovering high-frequency information by leveraging the sampling strategies of multiple ocean profilers. Our RFR is trained, validated, and tested on BGC-Argo and shipboard data to within %\sim%3\% accuracy, then applied to observations from two rapid-sampling Seagliders deployed during the Southern Ocean Glider Observations of the Submesoscale (SOGOS) experiment in 2019. This approach generates novel nitrate distributions at 50 times the horizontal resolution of the original BGC-Argo float data. Using the high-resolution RFR outputs, we identify signatures of nutrient injection into the upper ocean that coincide with enhanced stirring in a turbulent region downstream of the Southwest Indian Ridge. Relating these intermittent transport events to biological time series provides new observational evidence that small-scale stirring mediates additional nutrient drawdown and primary production in this region. We note that the Southern Ocean has outsized importance in the global carbon cycle (Gray 2024), such that constraining nutrient dynamics here will improve parameterization of larger ocean models. In our exploration of high-frequency nitrate variability in the Southern Ocean, RFR extends the scientific capabilities of publicly available BCG-Argo data and allows for deeper understanding of biogeochemical cycling at a more comprehensive set of scales. As a flexible approach that can be generalized to suit other multi-platform observing systems, RFR presents new opportunities to maximize value from existing datasets.

### Tutorial

Three main notebooks describe the (1) Random Forest model development, (2) application to ocean data and performance evaluation, followed by (3) scientific analysis of the output. 

Steps:


1. Random Forest Training: `scripts/Training_RandomForest.ipynb`

- This notebook demonstrates our machine learning approach to ocean nutrient prediction. We explore different variations of a Random Forest regressor using data from autonomous ocean profilers [see AIES paper for more detail]. 

- The process of model development includes three main phases: training, validation, and testing. The models are trained on different feature lists, then validated to screen for the best model parameters. We stress that additional cross-validation metrics are needed for observations with spatiotemporal biases (Stock et al. 2022).

- The testing phase uses withheld float data from the SOGOS BGC-Argo float to estimate the model error. We also compare performance with two well-established machine learning models, ESPER-Mixed and CANYON-B, which both rely on neural networks. 


<!-- 2. Random Forest Analysis: `scripts/Analysis_RandomForest.ipynb`

- Here, we

-  -->


## Code Directory


### Scripts

Most major functions are stored in modules, which are used to control processing functions that may be needed across scripts. The `mod_RFR.py` holds the important class objects used in `Training_RandomForest.ipynb`. We use the environment `mlsogos`, in the `binder` folder. 


- *Modules*: 

                mod_RFR       as rfr            Main random forest regression methods
                mod_MLV       as mlv            Mixed layer variability and wavelet
                
                mod_main      as sg             Data and main ancillary functions.
                mod_L3proc    as gproc          Used for xarray Datset processing of the level 3, 'L3' gridded glider product
                mod_DFproc    as dfproc         Used for pandas Dataframe processing during analysis
                mod_plot      as sgplot         Used to define common plotting parameters; can reproduce all paper figs




### Folders

- `scripts/` : code for analysis
- `data/` : float, ship, and glider data as downloaded
- `working-vars/` : calculated output variables from analysis
- `images/` : final output figures


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


- Description of glider data variables are in `Seaglider_DataGuide.pdf`
- Description of Argo float and ship are in first paper from SOGOS program: [Link to Dove et. al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JC017178)


The SOGOS data for Seagliders SG659 and SG660 can be accessed through \citet{balwada_2023_8361656}; DOI 10.5281/zenodo.8361656. Argo float data were collected and made freely available by the International Argo Program and the national programs that contribute to it (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program is part of the Global Ocean Observing System; Argo float data and metadata from Global Data Assembly Centre (\citet{argo2021_Argo}; DOI 10.17882/42182). Shipboard data were collected and made publicly available by the International Global Ship-based Hydrographic Investigations Program (GO-SHIP; http://www.go-ship.org/) and the national programs that contribute to it. The satellite altimetry data are freely available through the E.U. Copernicus Marine Environment Monitoring Service (CMEMS; DOI 10.48670/moi-00148), and the value-added FSLE product is provided by AVISO (https://www.aviso.altimetry.fr/en/data/products/value-added-products/fsle-finite-size-lyapunov-exponents.html; DOI 10.24400/527896/a01-2022.002). MODIS-Aqua satellite data are hosted by NOAA and provided through the NASA Ocean Biology Processing Group (https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1par08day.html). 


Global Ocean Gridded L 4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing:

- [Link to altimetry (ADT, EKE) data:](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/download
)
- [Altimetry product doi: (https://doi.org/10.48670/moi-00148)

- [Link to float data](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/Es-ESkVfIlpHhpFq7o5LTaoBtqv6pWj6rntxMyXieLEq8A?e=FeRRjs)

<!-- - [Argo ERDDAP Data Server](http://www.argodatamgt.org/Access-to-data/ERDDAP-data-server) -->

- [Link to ship data](https://uwnetid-my.sharepoint.com/:f:/g/personal/sangsong_uw_edu/ErLtPwS6pdZClgo0Flp9lq8Bz73FRmUlhR2zf329gDH-3w?e=hCzidh)
- [GO-SHIP I6 Cruise Line Bottle Data](https://cchdo.ucsd.edu/cruise/325020190403)


### Other Notes

Two phases of processing were developed externally in MATLAB: 

- Oxygen optode time response correction (Courtesy of Yui Takeshita, MBARI)
- ESPER-Mixed Prediction ([Carter et al. 2021](https://doi-org.offcampus.lib.washington.edu/10.1002/lom3.10461))
- CANYON-B Prediction ([Bittig et al. 2018](https://doi.org/10.3389/fmars.2018.00328))
- ACC Front locations courtesy of [Sauve et al. 2023](https://doi.org/10.1029/2023JC019815)

### Acknowledgments


This work is supported by NSF awards OCE-1756956 and OCE-1756882. SS and ARG are also supported by NASA award NNX80NSSC19K1252, the U.S. Argo Program through NOAA award NA20OAR4320271, NSF award OCE-2148434, and NSF’s Southern Ocean Carbon and Climate Observations and Modeling (SOCCOM) project through award OPP-1936222. PDL was supported by the NOAA grant NA19NES4320002 (Cooperative Institute for Satellite Earth System Studies, CISESS) at the University of Maryland/ESSIC. We thank Geoff Shilling and Craig Lee at APL for their efforts in reprocessing of the glider data. We also extend sincere thanks to Yuichiro Takeshita for offering his insights and code for processing the glider oxygen optode lags. We use colormaps obtained from the cmocean package ([Thyng et al. 2016]{https://doi.org/10.5670/oceanog.2016.66}). 

