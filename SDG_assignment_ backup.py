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
import seaborn as sns
import matplotlib.patches as patches
from scipy.stats import kstest
import scipy as stats
import statsmodels.api as stm
from scipy.stats import spearmanr
#from Tkinter import *
#from scipy.spatial.distance import mahalanobis

"""TO DO:
    - Main analysis hypothesis testing, Mahalanobis distance, choose correlation coefficient
    - Comment on synergy or trade-off
    - create bubblechart based on SDG data, combined with population size
    - Add countrycodes to dataframes next to country names 
    - Describe data types and why possible to compare"""

#%% Loading in data as panda dataframe (task 1)

"""use function to select most recent common year in both dataframes"""

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

#%% Main analysis (task 4)

"""Define H0: There is no significant relationship between the two datasets. 
    HA: there is a significant relationship between the two datasets!!
    
    low p-value means it is unlikely that the differences 
    between these groups came about by chance. 
    Your cutoff for rejecting the null hypothesis will be 0.05 – 
    that is, when there is a less than 5% chance that you would 
    see these results if the null hypothesis were true.
    The null hypothesis (H0) is that the distribution of 
    which your data is a sample is equal to the standard 
    normal distribution with mean 0, std deviation 1
    
    The Lilliefors test is a normality test based on the KS test
    It is used to test the H0 that the dates comes from a normally distributed population
    ksstat : float --> Kolmogorov-Smirnov test statistic with estimated mean and variance.
    pvalue : float --> If the pvalue is lower than some threshold, e.g. 0.05, then we can
    reject the Null hypothesis that the sample comes from a normal distribution


    The spearman function takes two real-valued samples as arguments 
    and returns both the correlation coefficient in the range between -1 and 1 
    and the p-value for interpreting the significance of the coefficient."""


#Lilliefors test on both datasets

lillie_ind12 = stm.stats.lilliefors(ind12_final['ind12data'], dist='norm', pvalmethod='table')
print('Lilliefors statistic test results on indicator 12 dataset (test statistic and P-value respectively):', lillie_ind12)
lillie_ind2 = stm.stats.lilliefors(ind2_final['ind2data'], dist='norm', pvalmethod='table')
print('Lilliefors statistic test results on indicator 2 dataset (test statistic and P-value respectively):', lillie_ind2)

print('For both datasets resulting P < 0.05, therefore the data is not considered to originate from a normally distributed dataset')
print('It is chosen to continue with the nonparametric Spearmans rank correlation coefficient ')

#Test homoscedasticity by observation scatter plot of the two datasets

x1 = ind12_final['ind12data'] 
y1 = ind2_final['ind2data']

fig = plt.figure()
plt.xlabel('DMC per capita (t)') 
plt.ylabel('Prevalence undernourishment (%)')

plt.plot(x1, y1, 'o', color = 'tomato', alpha=0.5)

plt.savefig('ScatterSDGs.png', bbox_inches='tight', dpi=400)

plt.show()
plt.close()

#Spearman correlation coefficient and p-value
#print results to console

coef, p = spearmanr(x1, y1)

print('Spearmans correlation coefficient: %.3f' % coef)

#Interpret the significance

alpha = 0.05

if p > alpha:
	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are correlated (reject H0) p=%.3f' % p)
    
#%%

"""Test absence of outliers with Mahalobinis distance"""

#The Mahalanobis distance is the distance between two points 
#in a multivariate space. It’s often used to find outliers in 
#statistical analyses that involve several variables

#merge two SDG dataframes into one new

# df_SDGs = ind2_final.merge(ind12_final, on='GeoAreaName')
# data = df_SDGs

# def MahalanobisDist(data, verbose=False):
#     covariance_matrix = np.cov(data, rowvar=False)
#     if is_pos_def(covariance_matrix):
#         inv_covariance_matrix = np.linalg.inv(covariance_matrix)
#         if is_pos_def(inv_covariance_matrix):
#             vars_mean = []
#             for i in range(data.shape[0]):
#                 vars_mean.append(list(data.mean(axis=0)))
#             diff = data - vars_mean
#             md = []
#             for i in range(len(diff)):
#                 md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

#             if verbose:
#                 print("Covariance Matrix:\n {}\n".format(covariance_matrix))
#                 print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
#                 print("Variables Mean Vector:\n {}\n".format(vars_mean))
#                 print("Variables - Variables Mean Vector:\n {}\n".format(diff))
#                 print("Mahalanobis Distance:\n {}\n".format(md))
#             return md
#         else:
#             print("Error: Inverse of Covariance Matrix is not positive definite!")
#     else:
#         print("Error: Covariance Matrix is not positive definite!")

# def is_pos_def(A):
#     if np.allclose(A, A.T):
#         try:
#             np.linalg.cholesky(A)
#             return True
#         except np.linalg.LinAlgError:
#             return False
#     else:
#         return False


# print("data:\n {}\n".format(data))

# MahalanobisDist(data, verbose=True)


# data = df_SDGs
# x = df_SDGs['ind12data']


# #create function to calculate Mahalanobis distance
# def mahalanobis(x=None, data=None, cov=None):

#     x_mu = x - np.mean(data)
#     if not cov:
#         cov = np.cov(data.values.T)
#     inv_covmat = np.linalg.inv(cov)
#     left = np.dot(x_mu, inv_covmat)
#     mahal = np.dot(left, x_mu.T)
#     return mahal.diagonal()

#df_SDGs.head()

#create new column in dataframe that contains Mahalanobis distance for each row
# df_SDGs['mahalanobis'] = mahalanobis(x=df_SDGs, data=df_SDGs[['ind2data', 'ind12data']])

#Choose for parametric (Pearson) or non-parametric correlation coefficient (Spearman)

#Trade-off or synergy between the two? ---> Comment on this!!!

# Expected negative correlation, however only above 2,5 default value
# Make separate plot with only countries above 2,5% prevalence to test this hypothesis


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

#%% Bubble plot based on country population size (task 5a)

# fig=plt.figure()

# p1 = sns.scatterplot(data=final_df, x="ind12data", y="ind2data", size="popdata", alpha=0.5, sizes=(10, 500), 
#                      legend = False)
# p1 = sns.scatterplot(data=df_bubblechart, x="ind12data", y="ind2data", size="popdata", 
#                       alpha=1, sizes=(10, 500), palette=sns.color_palette('dark'), 
#                       hue='GeoAreaName', legend = True)

# # COLOR LEGEND (10 items)
# col_lgd = plt.legend(df_popul['GeoAreaName'], loc='upper left', 
#                       bbox_to_anchor=(-0.1, -0.2), fancybox=True, shadow=True, ncol=3)

# plt.xlabel('Material consumption per capita (tonnes)')
# plt.ylabel('Prevalence undernourishment (%)')

# plt.show()       

# """Make sure the population dataframe has same amount of rows as SDG dataframes!!! """

# #Add labels to 10 countries with highest population size

# #fig.savefig('BC_country_population.png', bbox_inches = 'tight', dpi=400)

# plt.savefig('Bubble.png', bbox_inches='tight', dpi=400)

# plt.show()
# plt.close()

#%% Bubblechart without seaborn 

fig, ax = plt.subplots()

colours = ['blue', 'green', 'red', 'lightsteelblue','teal', 'plum', 'black', 'orange', 'lawngreen', 'olive']

p2= plt.scatter(final_df['ind12data'], final_df['ind2data'],
              s= final_df['popdata']/600,
              c = 'tomato',
              alpha=0.3)

for i in range(len(df_bubblechart['popdata'])):
      plt.annotate(df_bubblechart.GeoAreaName[i], (df_bubblechart.ind12data[i], (df_bubblechart.ind2data[i] + 0.2)))

# p2 = plt.scatter(df_bubblechart['ind12data'], df_bubblechart['ind2data'],
#               s= df_bubblechart['popdata']/700,
#               c = colours,
#               alpha=0.8)

#https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html

# COLOR LEGEND (10 items)
# col_lgd = plt.legend(df_bubblechart['GeoAreaName'], loc='upper left', 
#                        bbox_to_anchor=(-0.1, -0.2), fancybox=True, shadow=True, ncol=3)


plt.ylabel("Prevalence undernourishment (%)", size=10)
plt.xlabel("DMC per capita (t)", size=10)

plt.show()


#%% World map based on countries and colour indication to show whether SDGs are met (task 5b)

#Basic Earth Plot
path = gpd.datasets.get_path('naturalearth_lowres')
aarde = gpd.read_file(path)

"""change conditions to be more intuitive goals (e.g. prevalence <2.5% means goal is met)"""

def world_df(aarde):
    aarde2 = aarde.merge(ind12_final, how='left', left_on=['name'], right_on=['GeoAreaName'])
    aarde3 = aarde2.merge(ind2_final['ind2data'],
                          how='left', left_on=['name'],
                          right_on=ind2_final['GeoAreaName'])
    
    #create list of conditions
    conditions = [
        (aarde3['ind12data'] >= aarde3['ind12data'].median()) & (aarde3['ind2data'] >= aarde3['ind2data'].median()),
        (aarde3['ind2data'] >= aarde3['ind2data'].median()), 
        (aarde3['ind12data'] >= aarde3['ind12data'].median()),
        (aarde3['ind12data'] <= aarde3['ind12data'].median()) & (aarde3['ind2data'] <= aarde3['ind2data'].median())]
    
    
    # create list of values we want to assign for each condition
    values = ['green', 'orange', 'blue', 'red']
    
    # create new column and use np.select to assign values to it using list as argument
    aarde3['color'] = np.select(conditions, values, default='lightgrey')
    
    aarde3['category'] = ''
    
    # create list of values we want to assign for each condition
    values2 = ['both above', 'social above', 'env above', 'both below']
    
    #create new column and then use np.select to assign values to it using our lists as argument
    aarde3['category']=np.select(conditions, values2, default='missing value')
    
    #assign colours to the different SDGs
    #both SDGs above the median = purple
    #only social SDG (ind 2) above median = blue
    #only env SDG (ind 12) above median = green
    #both below median = red
    #missing = grey
    
    return(aarde3)

gpd_df = world_df(aarde)
values = ['green', 'orange', 'blue', 'red']
fig= plt.figure()

fig, ax = plt.subplots()
gpd_df.plot(column='category',
            categorical=True,
            legend=True,
            color=gpd_df['color'],
            missing_kwds={'color', 'lightgrey'})

# Creates a rectangular patch for each contaminant, using the colors above

values2 = ['both above', 'social above', 'env above', 'both below']
color_dict = {'both above':'green',
              'social above':'orange',
              'env above':'blue',
              'both below':'red',
              'missing':'lightgrey'}

patch_list =[]
for i in values2:
    label = i.capitalize()
    color = color_dict[i]
    patch_list.append(patches.Patch(facecolor=color, 
                                    label=label, 
                                    alpha=0.9, 
                                    linewidth=0.5, 
                                    edgecolor='black'))

# Creates a legend with the list of patches above.
plt.legend(handles=patch_list, fontsize=8, loc='lower left',
        bbox_to_anchor = (0.01,0.2), title_fontsize=10)
    

plt.savefig('Worldje.png', bbox_inches='tight', dpi=400)

plt.show()
plt.close()


#%% Exporting main results to text file (task 6)




# Export the following to a txt file, ensuring 
# that it can nicely be read in a text editor:
# a) the two selected SDG indicators and their 
# description,
# b) the correlation coefficient (including its 
# type),
# c) the p-value,
# d) the interpretation about if the correlation 
# is statistically significant at a level of 
# 0.01, 0.05, or 0.1 or not at all (write a 
# function for that), and
# e) an answer if there is rather a trade-off or a 
# synergy between the two indicators 
# (depending on positive or negative 
# correlations).


#%% Code optimization (task 7)

#Profile code, identify slow sections
#try making slow sections more efficient

#%% Creation radio button (task 8)

# Create a radio button for user interaction. Bring the pop-up window to the attention of the 
# user, i.e. bring it to the front or make it blinking. Make sure that it is clear to the user what 
# the selection is about. Automatically close the pop-up window after the selection so that 
# the user knows the selection is done.
# Note: Please keep the code without interaction in a comment so that one can easily switch 
# back. This can help you to work on your code and me to review it.



#%% Old code 

# SOCIAL INDICATOR: 2.1.1 Prevalence of undernourishment
# ENV INDICATOR: 12.2.2 Domestic material consumption (DMC), DMC per capita, DMC per GDP
# datafile: SN_ITK_DEFCN (nr of undernourish people in millions)
# datafile: SN_ITK_DEFC (prevalence of undernourishment in %)
# datafile: EN_MAT_DOMCMPC (DMC per capita) by type of raw material in tonnes)
# datafile: EN_MAT_DOMCMPG (DMC per GDP) by type of raw material in kg per constant 2010 US dollars
# datafile: EN_MAT_DOMCMPT (DMC) by type of raw material in tonnes

#create new dataframe with total material consumption per country in tonnes
# use 'inplace = true' to implement the change in the variable explorer 
# Material consumption per capita: 16 products sum up and only use this new total for further analysis

# use function to groupby countries, afterwards use function to select only those countries present in both dataframes
# use function to create new dataframe with only comparable countries 

#ind12pivot = ind12.pivot(index=["Indicator", "SeriesDescription", "GeoAreaName",], columns="Type of product", values="2017").resetindex() ind12pivotsum = ind12pivot.sum(axis=1)

# ind12_summed = ind12.groupby(['GeoAreaName']).sum() #how to keep the other columns from the dataframe??
# ind2_summed = ind2.groupby(['GeoAreaName']).sum()

# delete rows from ind12 dataframe not present in ind2 dataframe
# delete rows from ind2 dataframe not present in ind12 dataframe

# ind12_final = ind12_summed[ind12_summed.index.isin(ind2['GeoAreaName'])] 
# ind12_final.reset_index(level=0, inplace=True)
# ind2_final = ind2_summed[ind2_summed.index.isin(ind12_final['GeoAreaName'])]
# ind2_final.reset_index(level=0, inplace=True)

# ind2_final.sort_values('GeoAreaName')
# ind12_final.sort_values('GeoAreaName')

#kstest on dataset 

# kstest(ind2_final['ind2data'], 'norm')
# kstest(ind12_final['ind12data'], 'norm')

#ind12.groupby(['GeoAreaName'], as_index=True).agg({'2017': 'sum', 'Indicator': 'first', 'SeriesDescription': 'first'})

# fltr = df_pop_rename['Country code'].isin(ind12_final.GeoAreaCode)
# df_pop_rename = df_pop_rename[fltr]

# filtr = df1[].isin(df2.GeoAreaCode)
# df1 = df1[filtr]

# scatter plot with scatter() function
# transparency with "alpha"
# bubble size with "s"
# color the bubbles with "c"
# x2 = ind12_final['data'] 
# y2 = ind2_final['data']

# #merge to one dataframe and then create bubblechart 

# fig = plt.figure()
# plt.scatter(ind12_final['data'], ind2_final['data'],
#               s= df_pop_select['data']/500,
#               c = 'tomato',
#               alpha=0.3, data = df_pop_select['data'])
# plt.xlabel("Population size", size=12)
# plt.ylabel("Population size", size=12)


#final_df = pd.concat([ind2_final, ind12_final, df_pop_select], axis=1)

# TEXT annotations for 10 most populous countries
# for i in range(len(df_popul['popdata'])):
#      plt.annotate(final_df.country[i], (final_df.ind12data[i], (final_df.ind2data[i] + 0.2)))

# size_lgd = plt.legend(final_df['popdata'], loc='lower center', borderpad=1.6, prop={'size': 20},
#                       bbox_to_anchor=(0.5,-0.45), fancybox=True, shadow=True, ncol=5)

# df_pop_select['2017'] = pd.to_numeric(df_pop_select['2017'])
# df_pop_select.sort_values('GeoAreaName')

#df_popula = df_pop_select.sort_values(['popdata'], ascending=True)

# for ctype, data in gpd_df.category:
#     color= Palette[ctype]
#     data.plot(color=color,
#         ax=ax,
#         label=ctype,
#         legend=True)
    
# roadPalette = {'both':'purple',
#                'social':'blue',
#                'neither':'red',
#                'env':'green',
#                'missing':'lightgrey'}

# for ctype in gpd_df['category']:
#     color=roadPalette[ctype]
    
# ctype.plot(color=color,
#            ax=ax,
#            label=ctype)
# ax.legend(bbox_to_anchor=(1.0, .5), prop={'size':12})
