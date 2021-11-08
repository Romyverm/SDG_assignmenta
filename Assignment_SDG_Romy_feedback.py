# ASSIGNMENT A ROMY SUSAP 
# SOCIAL INDICATOR: 2.1.1 Prevalence of undernourishment
# ENV INDICATOR: 12.2.2 Domestic material consumption (DMC), DMC per capita, DMC per GDP
# datafile: SN_ITK_DEFCN (nr of undernourish people in millions)
# datafile: SN_ITK_DEFC (prevalence of undernourishment in %)
# datafile: EN_MAT_DOMCMPC (DMC per capita) by type of raw material in tonnes)
# datafile: EN_MAT_DOMCMPG (DMC per GDP) by type of raw material in kg per constant 2010 US dollars
# datafile: EN_MAT_DOMCMPT (DMC) by type of raw material in tonnes

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

#%% Loading in data as panda dataframe (task 1)
#indicator 2 = social
#indicator 12 = environmental

#df1 = pd.read_excel('SN_ITK_DEFCN.xlsx')
ind2 = pd.read_excel('SN_ITK_DEFC.xlsx', usecols = [2, 4, 6, 28]) 
ind12 = pd.read_excel('EN_MAT_DOMCMPC.xlsx', usecols = [2, 4, 6, 9, 28])
#df4 = pd.read_excel('EN_MAT_DOMCMPG.xlsx')
#df5 = pd.read_excel('EN_MAT_DOMCMPT.xlsx')

#%% data filtering (task 2)

#df2: set < x to specified minimum value and convert objects of year 2017 to floats 
#df3: set object to float for year 2017
ind2['2017'] = ind2['2017'].str.replace('<','').astype(float)
ind12['2017'] = pd.to_numeric(ind12['2017'])

'''
consider using a function to check for the most recent year, now it is hard coded
'''

   
#check                  
ind2.dtypes
ind12.dtypes

#%% sorting data and sanity checking (task 3)

#sort countries alphabetically 
ind2.sort_values('GeoAreaName')
ind12.sort_values('GeoAreaName')

#create new row that sums up the total values in tonnes 
#result will be one country row with total GDP value

#check for NaN values in dataframes, detect NaN values and drop these

#detect and drop NaN values in dataset indicator 2
count_nan_in_ind2 = ind2.isnull().sum().sum()
print ('Count of NaN in indicator 2: ' + str(count_nan_in_ind2))    
ind2.dropna(subset = ["2017"], inplace=True)


#detect and drop NaN values in dataset indicator 12
count_nan_in_ind12 = ind12.isnull().sum().sum()
print ('Count of NaN in df3: ' + str(count_nan_in_ind12))
ind12.dropna(subset = ["2017"], inplace=True)



#%% TO DO 

#create new dataframe with total material consumption per country in tonnes
# use 'inplace = true' to implement the change in the variable explorer 
# Material consumption per capita: 16 products sum up and only use this new total for further analysis

# use function to groupby countries, afterwards use function to select only those countries present in both dataframes
# use function to create new dataframe with only comparable countries 

ind12_pivot = ind12.pivot(index=["Indicator", "SeriesDescription", "GeoAreaName",], columns="Type of product", values="2017").reset_index()
ind12_pivotsum = ind12_pivot.sum(axis=1) 

ind12_summed = ind12.groupby(['GeoAreaName']).sum() #how to keep the other columns from the dataframe??
ind2_summed = ind2.groupby(['GeoAreaName']).sum()

#ind12.groupby(['GeoAreaName'], as_index=True).agg({'2017': 'sum', 'Indicator': 'first', 'SeriesDescription': 'first'})

# delete rows from ind12 dataframe not present in ind2 dataframe
# delete rows from ind2 dataframe not present in ind12 dataframe

ind12_final = ind12_summed[ind12_summed.index.isin(ind2['GeoAreaName'])] 
ind12_final.reset_index(level=0, inplace=True)
ind2_final = ind2_summed[ind2_summed.index.isin(ind12_final['GeoAreaName'])]
ind2_final.reset_index(level=0, inplace=True)

ind2_final.sort_values('GeoAreaName')
ind12_final.sort_values('GeoAreaName')

#sanity check on similarity between two datasets?

#%% Main analysis (task 4)

#H0: the higher the prevalence of undernourishment % (ind2), the lower the per capita material consumption (ind12)

#Kolmogorov Smirnov test on the two datasets
#two-sided: H0 is that the two distributions are identical, F(x)=G(x) for all x; the alternative is that they are not identical.

#kstest on dataset 
#kstest(rvs=, cdf=, args=(ind2_final['2017']), N=20, alternative='two-sided', mode='auto') 


#Test homoscedasticity by observation scatter plot of the two datasets

x1 = ind12_final['2017'] 
y1 = ind2_final['2017']

fig = plt.figure()
plt.xlabel('per capita mat consumption (tonnes)') 
plt.ylabel('% prevalence undernourishment')

plt.plot(x1, y1, 'o', color = 'darkred')


#Test absence of outliers with Mahalobinis distance 

#Choose for parametric or non-parametric correlation coefficient

#Trade-off or synergy between the two?
#comment on this


#%% Plot making (task 5)

# I'm quite behind on the assignment, and sorry that the code is not really intuitive yet!!



