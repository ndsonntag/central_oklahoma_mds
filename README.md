# Assessing the Applicability of Machine Learning Models in Predicting Potential SPC Severe Weather Watches in Central Oklahoma

Authors: Nathan Sonntag and Tyson Stewart

### Introduction

Within meteorological spaces, there has been an increasing amount of discussion as to how, if at all, machine learning (ML) models can be incorporated into research and forecasting operations. While most of this effort has come to fruition in the form of ML-enhanced numerical models (Lam et al. 2023; Price et al. 2024), research has been limited regarding the use of ML models to help with nowcasting operations. This project seeks to remedy some of these deficiencies by developing an ML model with the goal of predicting whether a severe weather watch will be issued for a given region or not. By doing this, we hope that this can serve as a potential proof of concept towards future models that can help forecasters nowcast areas of concern.

### Background

Prior literature on the usage of ML models in the meteorological space seems to mostly be consistent of two major categories. The first is the augmentation of numerical weather prediction (NWP). Traditionally, NWP involves taking a set of initial conditions (often referred to as “boundary conditions”) and performing a series of mathematical formulas and calculations to determine how any weather patterns will develop in the future (Waqas et al. 2024). This process, while validated across decades of use, is very computationally intensive, limiting the types of users and use-cases that it is effective for. One proposed way to combat this computational resource dilemma has been to create NWP models that are aided with ML processes. The rationale behind this being that ML models can use their ability to excel at pattern recognition to offload some of the raw computational work from NWP models. Several attempts at such a hybrid system have already been created with varying methodologies and results (Lam et al. 2023; Price et al. 2024; Bouallègue et al. 2024; Waqas et al. 2024).

The other major area of meteorological ML research has been in regard to the subject of whether ML models can supplementally assist with forecasting operations. Such research has looked at whether ML models can take atmospheric data from observations and numerical weather model analyses, and leverage pattern recognition to help forecast specific types of severe weather (Hill et al. 2020; Chkeir et al. 2023). While much of this research has been targeted on specific storm types, said research is broad in terms of study area scope (Hill et al. 2020; Prudden et al. 2020; Albu et al. 2022; Yao et al. 2022; Chkeir et al. 2023). There is seemingly a lack of research done on local to regional scale nowcasting ML models, a subject we seek to address. The goal of this project, therefore, is to create an ML model with data for a specific National Weather Service weather forecast office (WFO) region that attempts to predict whether a severe weather watch should/will be issued.

### Data

Data for this project will consist of two major parts. The first is a collection of all SPC MDs and watches for central Oklahoma from 2014-2024 filtered for severe weather only (excluding floods and winter weather events). For this study, central Oklahoma is centered on Oklahoma County, OK, and contains this and any county that shares a border with Oklahoma County. This data, while theoretically able to be pulled from the SPC directly, has been chosen to be acquired from the Iowa Environmental Mesonet’s (IEM) archival services for ease of use. Through IEM, we can obtain a shapefile containing all watches and MDs for our study region and time period and then use the product IDs included for each MD/watch to acquire and insert the relevant textual data through IEM’s SPC text archive. This textual data includes information on weather type, watch probability, and product type, which we can use to filter and group each MD/watch into their respective category.
The second group of data to be acquired is the atmospheric data that will be used to train the model, test, and validate the model in conjunction with the SPC data. For our purposes we have elected to use the Rapid Refresh (RAP) model analysis, as it consists of hourly, high-definition atmospheric data for North America. RAP analysis also includes many of the variables previously established in prior literature to be relevant to ML analysis of severe weather (Hill et al. 2020), further justifying its use case here. Given the hourly interval, RAP analysis data for the nearest hour prior to the MD or watch will be utilized. This is to avoid “overshooting” of the environmental conditions that might happen if we were to choose the data after the MD was issued.


### Methodology
#### a. Data Preprocessing

During the filtering of SPC products over central Oklahoma, products were also filtered by product type (either as an MD or as a watch). This was done to create the two ‘labels’ that the ML model will try to predict based on the average environmental variables obtained from RAP analysis. Once the climatology of SPC products was obtained and filtered, each product was environmentally analyzed using RAP analysis obtained from the National Center for Environmental Information Thredds server (https://www.ncei.noaa.gov/thredds/catalog.html). For each product, a total of 14 variables were obtained directly from the RAP analysis itself or calculated using MetPy functions (https://unidata.github.io/MetPy/latest/index.html). The selection of the 14 variables follows closely to those used in a severe weather prediction Random Forest (RF) model produced by Hill et al. (2020) and are listed in Table 1. Each of these variables were averaged over all grid points within a 60-km radius of downtown Oklahoma City and filtered to not include any grid points that have a composite reflectivity of 30 dBz or greater. Furthermore, if a file was either missing variables or missing entirely from the Thredds database, that product was filtered out of the final dataset for ML training and testing. 

***Table 1: Environmental Variables and Acronyms***

| Environmental Variable                     | Acronym       |
|--------------------------------------------|---------------|
|     2-m Temperature                        |     t2m       |
|     2-m Specific Humidity                  |     q2m       |
|     2-m Relative Humidity                  |     rh2m      |
|     10-m U-Component Wind                  |     u10       |
|     10-m V-Component wind                  |     v10       |
|     10-m Wind Speed                        |     uv10      |
|     Mean Sea-Level Pressure                |     mslp      |
|     Precipitable Water                     |     pwat      |
|     Lifted Condensation Pressure Level     |     lcl       |
|     Surface-Base CAPE                      |     cape      |
|     Surface-Based CIN                      |     cin       |
|     0-6 km Bulk Shear                      |     shr0_6    |
|     0-1 km Storm Relative Helicity         |     srh0_1    |
|     0-3 km Storm Relative Helicity         |     srh0_3    |

#### b.	Random Forest Testing and Training

Given the tabular nature of the data collected, an RF model was determined to be the best predictive tool. RF models have been used to great success in the past in the prediction of severe weather conditions (Hill et al. 2020, Loken et al. 2022). As a result, it was determined that this model would likely also perform well for similar purposes in this research. As an initial model, all environmental variables, including month and time of product occurrence, will be used. (Exact details to follow as model development continues)

### Results
#### a. Data Validation

#### b. Environmental Comparison Between Mesoscale Discussions and Watches

#### c. Random Forest Model Analysis

***Table 2: Base Model Classification Report***
|              | Precision | Recall    | F1-Score  | Support   |
| MDs          | 0.72      | 0.79      | 0.75      | 48        |
| Watchs       | 0.77      | 0.69      | 0.73      | 49        |
|              |           |           |           |           |
| Accuracy     |           |           | 0.74      | 97        |
| Macro Avg    | 0.74      | 0.74      | 0.74      | 97        |
| Weighted Avg | 0.75      | 0.74      | 0.74      | 97        |


***Table 3: Optimized Model Classification Report***
|              | Precision | Recall    | F1-Score  | Support   |
| MDs          | 0.76      | 0.79      | 0.78      | 48        |
| Watchs       | 0.79      | 0.76      | 0.77      | 49        |
|              |           |           |           |           |
| Accuracy     |           |           | 0.77      | 97        |
| Macro Avg    | 0.77      | 0.77      | 0.77      | 97        |
| Weighted Avg | 0.77      | 0.77      | 0.77      | 97        |


***Table 4: Final Tested Model Average Classification Report***
|              | Precision | Recall    | F1-Score  | Support   |
| MDs          |           |           |           | 105       |
| Watchs       |           |           |           | 91        |
|              |           |           |           |           |
| Accuracy     |           |           |           | 196       |
| Macro Avg    |           |           |           | 196       |
| Weighted Avg |           |           |           | 196       |

### Conclusion



### References
Albu, A.-I., G. Czibula, A. Mihai, I. G. Czibula, S. Burcea, and A. Mezghani, 2022: NeXtNow: A Convolutional Deep Learning Model for the Prediction of Weather Radar Data for Nowcasting Purposes. Remote sensing, 14, 3890–3890, https://doi.org/10.3390/rs14163890.

Bouallègue, Z., and Coauthors, 2024: The rise of data-driven weather forecasting: A first statistical assessment of machine learning-based weather forecasts in an operational-like context. Bulletin of the American Meteorological Society, 105, https://doi.org/10.1175/bams-d-23-0162.1.

Chkeir, S., A. Anesiadou, A. Mascitelli, and R. Biondi, 2023: Nowcasting extreme rain and extreme wind speed with machine learning techniques applied to different input datasets. Atmospheric Research, 282, 106548, https://doi.org/10.1016/j.atmosres.2022.106548.

Lam, R., and Coauthors, 2023: Learning skillful medium-range global weather forecasting. Science, 382, https://doi.org/10.1126/science.adi2336.

Loken, E. D., A. J. Clark, and A. McGovern, 2022: Comparing and interpreting differently designed random forests for next-day severe weather hazard prediction. Wea. Forecasting, 37, 871–899, https://doi.org/10.1175/WAF-D-21-0138.1.

Waqas, M., U. W. Humphries, B. Chueasa, and A. Wangwongchai, 2024: Artificial Intelligence and Numerical Weather Prediction Models: A Technical Survey. Natural Hazards Research, https://doi.org/10.1016/j.nhres.2024.11.004.

Price, I., and Coauthors, 2024: Probabilistic weather forecasting with machine learning. Nature, 637, https://doi.org/10.1038/s41586-024-08252-9.

Prudden, R., S. V. Adams, D. Kangin, N. H. Robinson, S. V. Ravuri, S. Mohamed, and A. Arribas, 2020: A review of radar-based nowcasting of precipitation and applicable machine learning techniques. arXiv (Cornell University), https://doi.org/10.48550/arxiv.2005.04988.

Yao, S., H. Chen, E. Thompson, and R. Cifelli, 2022: An Improved Deep Learning Model for High-Impact Weather Nowcasting. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 7400–7413, https://doi.org/10.1109/jstars.2022.3203398.
