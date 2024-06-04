# Random Forest Model Development

Paper submitted to to AMS Artifical Intelligence in Earth Sciences (AIES) journal May 2024, pending review. - [Link to pre-print](https://doi.org/10.22541/essoar.171707849.91867565/v1)


## Notebooks

To see the main steps of RFR model development, see:

- [Main Training Notebook:](scripts/Training_RandomForest.ipynb) `./Training_RandomForest.ipynb`

    - More analysis and troubleshooting figures in `./Analysis_RandomForest.ipynb`


To see how the input data was processed, see:

- [BGC-Argo Float Processing Notebook:](scripts/SG_Float_Processing.ipynb) `./SG_Float_Processing.ipynb`
- [Seaglider Processing Notebook:](scripts/SG_Glider_Processing.ipynb) `./SG_Glider_Processing.ipynb`
- [GO-SHIP Processing Notebook:](scripts/SG_Ship_Processing.ipynb) `./SG_Ship_Processing.ipynb`
Updated Jun 04 2024

<!-- ## Code Directory -->

## Modules

Major functions and custom classes are stored in modules. 

The `mod_RFR.py` holds the important class objects used in `Training_RandomForest.ipynb`. 


- *Modules*: 

                mod_RFR       as rfr            Main random forest regression methods
                mod_MLV       as mlv            Mixed layer variability and wavelet
                
                mod_main      as sg             Data and main ancillary functions.
                mod_L3proc    as gproc          Used for xarray Datset processing of the level 3, 'L3' gridded glider product
                mod_DFproc    as dfproc         Used for pandas Dataframe processing during analysis
                mod_plot      as sgplot         Used to define common plotting parameters; can reproduce all paper figs


## Naming Conventions

- *Glider:*    

                dav_659                         Profiles-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                profid                          Unique name for each profile

- *Floats:*    

                dav_659                         Profile-averaged metrics, incl. MLD
                df_659                          Flattened dataframe
                wmoid                           Unique name for each float WMO#
                profid                          Unique name for each profile


## Other Notes

Two phases of processing were developed externally in MATLAB: 

- Oxygen optode time response correction (Courtesy of Yui Takeshita, MBARI)
- ESPER-Mixed Prediction ([Carter et al. 2021](https://doi-org.offcampus.lib.washington.edu/10.1002/lom3.10461))
- CANYON-B Prediction ([Bittig et al. 2018](https://doi.org/10.3389/fmars.2018.00328))
- ACC Front locations courtesy of [Sauve et al. 2023](https://doi.org/10.1029/2023JC019815)