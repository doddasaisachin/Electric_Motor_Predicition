import pandas as pd

import numpy as np

df=pd.read_csv('data_set/temperature_data.csv')
df_copy=df.copy()

df_copy.drop('profile_id',inplace=True,axis=1)

# def correlated(dataframe):
#     relations=[]
#     corr_df=dataframe.corr()
#     for i in dataframe.columns:
#         for j in dataframe.columns:
#             if i!=j:
#                 if (corr_df[i][j]>=0.90) or (corr_df[i][j]<=-0.90):
#                     if set((i,j)) not in relations:
#                         relations.append(set((i,j)))
#     return relations

# corr_set=correlated(df_copy)
# corr_set

df_copy.drop(['stator_winding','stator_yoke'],axis=1,inplace=True)

Y=df_copy.pop('motor_speed')
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaled_df=pd.DataFrame(scaler.fit_transform(df_copy),columns=df_copy.columns)

scaled_df.head()

import pickle

pickle.dump(scaler,open('scaler.pkl','wb'))





X=df_copy.values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.989)

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape

from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor()

import warnings
warnings.filterwarnings('ignore')

reg.fit(xtrain,ytrain)

from sklearn.metrics import r2_score

# pred=reg.predict(xtest[:100000])

# r2_score(ytest[:100000],pred)*100

# reg.score(xtest[-100000:],ytest[-100000:])*100

pickle.dump(reg,open('model.pkl','wb'))
