#!/usr/bin/env python
# coding: utf-8

# # Walmart Data Analysis

# In[117]:


# Importing required packages and data for data analysis 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[118]:


# reading data from csv using the pandas 
df_features=pd.read_csv(r"C:\Users\YOGESH\OneDrive\Desktop\Phase_1\features.csv")
df_store = pd.read_csv(r"C:\Users\YOGESH\OneDrive\Desktop\Phase_1\features.csv")
df_train=pd.read_csv(r"C:\Users\YOGESH\OneDrive\Desktop\Phase_1\train.csv")


# # EDA AND DATA CLEANING 

# In[119]:


# first 5 rows of data(features)
df_features.head()


# In[120]:


# first 5 rows of data(store)
df_store.head()


# In[121]:


# Shape of the dataframe(features)
df_features.shape


# In[122]:


# Shape of the dataframe(store)
df_store.shape


# In[123]:


# first 5 rows of dataset(train)
df_train.head()


# In[124]:


# The shape of dataframe(train)
df_train.shape


# This info method helps us to get total entries, column names and their dtypes, and Non null entries in the columns

# In[125]:


# Dataset info (features)
df_features.info()


# In[126]:


# Dataset info (store)
df_store.info()


# In[127]:


# Dataset info (train)
df_train.info()


# In[128]:


# Describe method gives all the stats about the data frame


# In[129]:


df_features.describe()


# In[130]:


df_store.describe()


# In[131]:


df_train.describe()


# In[132]:


# Now we are going to check null entries in the dataframes


# In[133]:


df_features.isnull().sum()


# In[134]:


df_store.isnull().sum()


# In[135]:


df_train.isnull().sum()


# In[136]:


# We found that features dataframe contains the null values so we can use fillna method to fill with zeros 


# In[137]:


df_features = df_features.fillna(0)


# In[138]:


# We cleared all the null values in the data frame.
df_features.isnull().sum()


# In[139]:


# Here we are merging our dataset stores and features
df_new = df_features.merge(df_store, how = 'inner', on = 'Store')
df_new.head()


# In[140]:


# Let's inspect new dataframe
df_new.info()


# In[141]:


df_new.describe()


# In[142]:


# Here we can see that our Date column is object, since it's a object we can't use it to it's fullest so lets decompose it. 


# In[143]:


# Here we are importing the datatime package
from datetime import datetime


# In[144]:


df_new['Date']=pd.to_datetime(df_new['Date'])


# In[145]:


# Since the same column is present in the train data we need to convert it to the datetime
df_train['Date']=pd.to_datetime(df_train['Date'])


# In[146]:


# Here we are creating two new columns such as week and year for easier analysis  
df_new['week']=df_new.Date.dt.isocalendar().week

df_new['year']=df_new.Date.dt.isocalendar().year


# In[147]:


df_new.head()


# In[148]:


df_new.info()


# In[149]:


# Here we are joining our dataset train and df_new from the earlier for the whole dataset. Since having the whole can help us to identify the problem and also aids us in getting the correct insights.

df = df_train.merge(df_new,how='inner', on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)


# In[150]:


# Retrieving the top five rows
df.head()


# In[151]:


# For getting the information about the dataframe.
df.info()


# Since we have seen that Markdown1, markdown2, markdown3, markdown4 and markdown5 columns contains more number of zeros. 
# We can consider it for dropping from the table. Let's wait until the end of analysis, if we can use it or not.

# In[152]:


# we are going to see if there are duplicate values 
df.duplicated().sum()


# In[153]:


# For our convinience we are going to rename the IsHoliday as Holiday
df=df.rename(columns={"IsHoliday":'Holiday'})
df.info()


# In[154]:


df.columns


# In[155]:


# as we disscussed, we have to see if we can drop markdown columns or not, for that we use correlation
df.corr()


# since the markdown values are mostly negitive and zero it is evident that they do not contribute to weekly sales or any other attributes.so we drop them. 
# 

# In[156]:


df=df.drop('MarkDown1', axis=1)
df=df.drop('MarkDown2', axis=1)
df=df.drop('MarkDown3', axis=1)
df=df.drop('MarkDown4', axis=1)
df=df.drop('MarkDown5', axis=1)
df.info


# In[157]:


df.info()


# ## Visualizations

# In[158]:


# Let's find out how many unique deptartments and stores present in our dataframe
df.Store.unique()

# We found that we have 45 unique stores in the dataframe


# In[159]:


# This plot shows the distribution of different stores spread across our data
df.Store.hist(color='red')


# In[160]:


df.hist(figsize=(30,30))


# In[161]:


# Let's find out the unique departments in the dataframe.
df.Dept.unique()
# We found that there are 65 different departments in the dataset


# In[162]:


# Let's find that how many years weekly sales does our df contain.


# In[163]:


df['year'].value_counts().plot.bar(color='red')


# In[164]:


# In the above graph we get to know that our data contains three years weekly sales 


# In[165]:


# The scatter plot points the weekly sales of different stores  
plt.figure(figsize=(10,10))
plt.scatter(df['Store'],df['Weekly_Sales'], c=df.index, cmap='inferno')
plt.ylabel('Weekly_Sales')
plt.xlabel('Store')

plt.show()


# In[166]:


# The scatter plot points the weekly sales of different dept  
plt.figure(figsize=(10,10))
plt.scatter(df['Dept'],df['Weekly_Sales'])
plt.ylabel('Weekly_Sales')
plt.xlabel('Dept')


# In[167]:


#Here we are getting the weekly sales by store
plt.figure(figsize=(25,10))

sns.barplot(x='Store', y='Weekly_Sales', data=df)


# In[168]:


#Here we are getting the weekly sales by department
plt.figure(figsize=(25,10))

sns.barplot(x='Dept', y='Weekly_Sales', data=df)


# In[169]:


# Lets find the correlation among the features in the dataset
plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),annot =True)



# In[177]:


# Here we are getting the sales by yearly
sales_2010 = df[df['year']==2010]['Weekly_Sales'].groupby(df['week']).mean()
sales_2011 = df[df['year']==2011]['Weekly_Sales'].groupby(df['week']).mean()
sales_2012 = df[df['year']==2012]['Weekly_Sales'].groupby(df['week']).mean()
#Plot a line plot with week on x axis and sales for that particular week of the filtered year in y axis
sns.lineplot(sales_2010.values,color='red')
sns.lineplot(sales_2011.values,color='green')
sns.lineplot(sales_2012.values,color='blue')


# In[ ]:


sns.distplot(df['Weekly_Sales'],color='red')


# In[179]:


fuel_2010 = df[df['year']==2010]['Fuel_Price'].groupby(df['week']).mean()
fuel_2011 = df[df['year']==2011]['Fuel_Price'].groupby(df['week']).mean()
fuel_2012 = df[df['year']==2012]['Fuel_Price'].groupby(df['week']).mean()


# In[181]:


sns.lineplot(fuel_2010.values,color='red')
sns.lineplot(fuel_2011.values,color='grey')
sns.lineplot(fuel_2012.values,color='black')


# In[182]:


#Box plot between the size and type of stores
sns.boxplot(data = df, x = 'Type', y = 'Size')


# In[183]:


# Lets check the temperature, fuel prices and unemployment over the weekly sales.
Fuel_price_over_sales = pd.pivot_table(df, values= "Weekly_Sales",index= "Fuel_Price")


# In[184]:


Fuel_price_over_sales.plot()


# In[185]:


temperature_over_sales = pd.pivot_table(df, values= "Weekly_Sales",index= "Temperature")


# In[186]:


temperature_over_sales.plot()


# In[187]:


unemployment_over_sales = pd.pivot_table(df, values= "Weekly_Sales",index= "Unemployment")


# In[188]:


unemployment_over_sales.plot()


# In[ ]:




