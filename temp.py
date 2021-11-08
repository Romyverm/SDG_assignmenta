# ASSIGNMENT A ROMY SUSAP 
# SOCIAL INDICATOR: 2.1.1 Prevalence of undernourishment
# ENV INDICATOR: 12.2.2 Domestic material consumption (DMC), DMC per capita, DMC per GDP
# datafile: SN_ITK_DEFC (prevalence of undernourishment in %)
# datafile: EN_MAT_DOMCMPC (DMC per capita) by type of raw material in tonnes)
#indicator 2 = social
#indicator 12 = environmental

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.patches as patches
# from scipy.stats import kstest
# import scipy as stats
import statsmodels.api as stm
from scipy.stats import spearmanr

#from scipy.spatial.distance import mahalanobis

"""TO DO:
    - Main analysis hypothesis testing, Mahalanobis distance, choose correlation coefficient
    - Comment on synergy or trade-off
    - create bubblechart based on SDG data, combined with population size
    - Add countrycodes to dataframes next to country names """

#%% Loading in data as panda dataframe (task 1)

"""change to function to select most recent common year in both dataframes"""

ind2 = pd.read_excel('SN_ITK_DEFC.xlsx', usecols = [2, 4, 5, 6, 28]) 
ind12 = pd.read_excel('EN_MAT_DOMCMPC.xlsx', usecols = [2, 4, 5, 6, 9, 28])

#%% data filtering (task 2)

#df2: set < x to specified minimum value and convert objects of year 2017 to floats 
#df3: set object to float for year 2017
ind2['2017'] = ind2['2017'].str.replace('<','').astype(float)
ind12['2017'] = pd.to_numeric(ind12['2017'])
   
#check                  
ind2.dtypes
ind12.dtypes

#%% sorting data and sanity checking (task 3)

"""use loop here again"""

#check for NaN values in dataframes, detect NaN values and drop these

"""create loop here for similar activity on multiple datasets"""

#detect and drop NaN values in dataset indicator 2
count_nan_in_ind2 = ind2.isnull().sum().sum()
print ('Count of NaN in indicator 2: ' + str(count_nan_in_ind2))
ind2.dropna(subset = ["2017"], inplace=True)

#detect and drop NaN values in dataset indicator 12
count_nan_in_ind12 = ind12.isnull().sum().sum()
print ('Count of NaN in df3: ' + str(count_nan_in_ind12))
ind12.dropna(subset = ["2017"], inplace=True)

#%% Filtering data (task 3)

#create new row that sums up the total values in tonnes 
#result will be one country row with total GDP value in tonnes

# ind12_pivot = ind12.pivot(index=["Indicator", "SeriesDescription", "GeoAreaName",], columns="Type of product", values="2017").reset_index()
# ind12_pivotsum = ind12_pivot.sum(axis=1) 

ind12_summed = ind12.groupby(['GeoAreaName']).sum()

#ind12.groupby(['GeoAreaName','GeoAreaCode'])['2017'].sum()

ind2_summed = ind2.groupby(['GeoAreaName']).sum()

# delete rows from ind12 dataframe not present in ind2 dataframe
# delete rows from ind2 dataframe not present in ind12 dataframe

ind12_final = ind12_summed[ind12_summed.index.isin(ind2['GeoAreaName'])] 
ind12_final.reset_index(level=0, inplace=True)

ind2_final = ind2_summed[ind2_summed.index.isin(ind12_final['GeoAreaName'])]
ind2_final.reset_index(level=0, inplace=True)

ind2_final.sort_values('GeoAreaName')
ind12_final.sort_values('GeoAreaName')

ind12_final = ind12_final.rename({'2017': 'ind12data'}, axis=1)
ind2_final = ind2_final.rename({'2017': 'ind2data'}, axis=1)

# filtr = df1[].isin(df2.GeoAreaCode)
# df1 = df1[filtr]

#sanity check on similarity between two datasets?

#%% Data import country population size (task 5a)

#Make bubblechart based on country population size --> total population, both sexes
#Annotations indicate country names for the 10 most populous countries

#read in population datafile
df_pop = pd.read_excel('WPP2019.xlsx', usecols = [2, 4, 74], skiprows=16) 

df_pop_rename = df_pop.rename({'Region, subregion, country or area *': 'GeoAreaName'}, axis=1)
df_pop_rename['2017'] = pd.to_numeric(df_pop_rename['2017'], errors='coerce')

df_pop_summed = df_pop_rename.groupby(['GeoAreaName']).sum()

#Use loop to drop columns if not index, region or year 2017? 

#select only countries present in indicator dataframes

df_pop_select = df_pop_summed[df_pop_summed.index.isin(ind2_final['GeoAreaName'])] 
df_pop_select.reset_index(level=0, inplace=True)

#rename year 2017 column to popdata

df_pop_select = df_pop_select.rename({'2017': 'popdata'}, axis=1)

filtr = ind12_final['GeoAreaName'].isin(df_pop_select.GeoAreaName)
ind12_final = ind12_final[filtr]

filtr2 = ind2_final['GeoAreaName'].isin(df_pop_select.GeoAreaName)
ind2_final = ind2_final[filtr2]

#join three dataframes into one new dataframe 

final_df = pd.merge(pd.merge(ind2_final,ind12_final,on='GeoAreaName'),df_pop_select,on='GeoAreaName')

#create new dataframe for annotations 10 most populous countries add label in graph
"""create additional dataframe for label plot"""

df_popul = df_pop_select.nlargest(10,'popdata')
df_bubblechart = pd.merge(pd.merge(ind2_final,ind12_final,on='GeoAreaName'),df_popul,on='GeoAreaName')

#%% Data import country population size (task 5a)

#Make bubblechart based on country population size --> total population, both sexes
#Annotations indicate country names for the 10 most populous countries

#read in population datafile
df_pop = pd.read_excel('WPP2019.xlsx', usecols = [2, 4, 74], skiprows=16) 

df_pop_rename = df_pop.rename({'Region, subregion, country or area *': 'GeoAreaName'}, axis=1)
df_pop_rename['2017'] = pd.to_numeric(df_pop_rename['2017'], errors='coerce')

df_pop_summed = df_pop_rename.groupby(['GeoAreaName']).sum()

#Use loop to drop columns if not index, region or year 2017? 

#select only countries present in indicator dataframes

df_pop_select = df_pop_summed[df_pop_summed.index.isin(ind2_final['GeoAreaName'])] 
df_pop_select.reset_index(level=0, inplace=True)

#rename year 2017 column to popdata

df_pop_select = df_pop_select.rename({'2017': 'popdata'}, axis=1)

filtr = ind12_final['GeoAreaName'].isin(df_pop_select.GeoAreaName)
ind12_final = ind12_final[filtr]

filtr2 = ind2_final['GeoAreaName'].isin(df_pop_select.GeoAreaName)
ind2_final = ind2_final[filtr2]

#join three dataframes into one new dataframe 

final_df = pd.merge(pd.merge(ind2_final,ind12_final,on='GeoAreaName'),df_pop_select,on='GeoAreaName')

#create new dataframe for annotations 10 most populous countries add label in graph
"""create additional dataframe for label plot"""

df_popul = df_pop_select.nlargest(10,'popdata')
df_bubblechart = pd.merge(pd.merge(ind2_final,ind12_final,on='GeoAreaName'),df_popul,on='GeoAreaName')

#%% simple plot from internet


#Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

#fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# plt.show()
# plt.close()
