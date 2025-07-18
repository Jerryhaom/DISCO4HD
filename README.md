# DISCO4HD <br>
This package provides measurements of Distance of Covariance (DISCO) <br>
DISCO quantifies homeostatic dysregulation by comparing biomarker covariance between target population and young reference population. <br>
The example data was downloaded from the National Health and Nutrition Examination Survey (NHANES). 

# INSTALL <br>
install.packages("devtools") <br>
devtools::install_github("Jerryhaom/DISCO4HD") <br>

# Examples
#Load sample data <br>
data(NHANES4) <br>
#impute missing data <br>
NHANES4=imputeMissings::impute(NHANES4) <br>
#Define biomarkers  <br>
biomarkers <- c("albumin", "alp", "creat", "glucose_mmol", "lymph", "mcv") <br>

#Create young reference (age ≤ 30) <br>
ref_young <- subset(NHANES4, age <= 30) <br>
#Calculate DISCO (single-threaded R, not recommended) <br>
result <- cal_disco(NHANES4, biomarkers, ref_young) <br>
#Parallel R implementation <br>
result_parallel <- cal_disco(NHANES4, biomarkers, ref_young, parallel = TRUE) <br>
#C++ implementation (recommended for large datasets) <br>
result_cpp <- cal_disco(NHANES4, biomarkers, ref_young, cpp = TRUE) <br>

# Implementation with Python code
pip install Py/dist/DISCO4HD-1.0-cp310-cp310-linux_x86_64.whl

import DISCO4HD  <br>
from DISCO4HD import cal_disco  <br>
help(help(DISCO4HD.cal_disco))  <br>
#for demo example data (randomly generated) <br>
DISCO4HD.example() <br>

#load NHANES data



# Citation <br>
Meng Hao et al. Distance of covariance (DISCO), a novel measure of network homeostatic dysregulation, reveals organ system interconnections underlying mortality and disease risk. doi: 10.1101/2025.05.06.25327108

