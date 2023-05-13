#!/usr/bin/env python
# coding: utf-8

# ## Pipeline

# In[13]:


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)
sns.set(rc={'figure.figsize':[7,7]},font_scale=1.2)


# In[3]:


df=pd.read_pickle('processed_2.pickle') 
df_train=pd.read_csv('training_data.csv')
df_test=pd.read_csv('testing_data.csv')


# In[6]:


class feature_engineering(BaseEstimator,TransformerMixin) : 
    def __init__(self,date_format='%Y-%m-%d %H:%M:%S',season_dict={1:'spring',2:'summer',3:'fall',4:'winter'},
                                    weather_dict={1:'Clear',2:'Cloudy',3:'Snow'}) :
        self.date_format=date_format
        self.season_dict=season_dict
        self.weather_dict=weather_dict
        
    def fit(self,X,y=None) : 
        return self 
    
    def transform(self,X,y=None) :  
        X_copy=X.copy()
        X_copy['datetime']=pd.to_datetime(X_copy['datetime'],format=self.date_format,errors='coerce')
        X_copy['year']=X_copy['datetime'].dt.year
        X_copy['month_name'] = X_copy['datetime'].dt.month_name()
        X_copy['day_of_week'] = X_copy['datetime'].dt.day_name() 
        X_copy['hour'] = X_copy['datetime'].dt.hour  
        X_copy.drop(columns=['datetime'], inplace=True) 
        X_copy['weather']=X_copy['weather'].replace(self.weather_dict)
        X_copy['season']=X_copy['season'].replace(self.season_dict) 
        X_copy['is_rush_hour']=X_copy['hour'].isin([17,18,8,19,16,7,9]).astype('category')
        X_copy['is_weekend']=X_copy['day_of_week'].isin(['Saturday','Sunday']).astype('category')
        
        return X_copy


# In[7]:


class select_features(BaseEstimator,TransformerMixin) : 
    def __init__(self,num_features=['temp','humidity','windspeed'],cat_features=['season','holiday', 'workingday', 'weather','hour' ,'month_name', 'day_of_week','is_rush_hour','is_weekend']) : 
        self.num_features=num_features
        self.cat_features=cat_features
    
    def fit(self,X,y=None) : 
        return self 
    
    def transform(self,X,y=None) : 
        X=X[self.num_features+self.cat_features]
        for col in self.num_features : 
            X[col]=X[col].astype('float64')
        for col in self.cat_features : 
            X[col]=X[col].astype('category')
        return X


# In[8]:


add_new_features=feature_engineering()


# In[9]:


pick_features=select_features()


# In[14]:


preprocessor=joblib.load('preprocessor.pickle')


# **Now you can use this pipeline directly on raw training and testing datasets** 
# 
# **This pipeline prepare raw data directly to fed to machine learning model enjoy!!**

# In[ ]:




