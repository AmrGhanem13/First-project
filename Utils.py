#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Major
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn_features.transformers import DataFrameSelector


# In[2]:


df=pd.read_csv( os.path.join(os.getcwd(),'Files','housing.csv'))
## replace <1H OCEAN
df['ocean_proximity']=df['ocean_proximity'].replace("<1H OCEAN","1H OCEAN")
df['ocean_proximity'].unique()

## Try Feature Extraction
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedroms_per_rooms'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

## Splitting To Feature & Target
X=df.drop(columns=['median_house_value'],axis=1)
y=df['median_house_value']
X_train, X_test, y_train, y_test =train_test_split(X,y,shuffle=True,test_size=0.15,random_state=42)

## Separating Categorical & Numerical Features
num_col=[col for col in X_train.columns if X_train[col].dtype in ['float32','float64','int32','int64'] ]
cat_col=[col for col in X_train.columns if X_train[col].dtype not in ['float32','float64','int32','int64'] ]





## Pipilining Numerical Features
num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(num_col)),    ## select only these columns
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())])

## categorical pipeline
categ_pipeline = Pipeline(steps=[
            ('selector', DataFrameSelector(cat_col)),    ## select only these columns
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OHE', OneHotEncoder(sparse_output=False))])

##Concatenate Two Piplines
total_pip=FeatureUnion(transformer_list=[
                                   ('num_pip',num_pipeline),
                                    ('cat_pip',categ_pipeline)
])

X_train_final=total_pip.fit_transform(X_train)




def preprocess_new(X_new):
  ## This func Will process New instance before entering the Model,at the same order of the features(all Numerical except the last one is categorical)
  ##returns processed features which will be ready to be used by the model
     return total_pip.transform(X_new)






