# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:38:50 2021

@author: anton
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import datetime

#Ler ambos os ficheiros CSV
df = pd.read_csv("training_data.csv", encoding = 'latin-1')
df_final = pd.read_csv("test_data.csv", encoding = 'latin-1')

#Remover colunas sem informação relevante
df = df.drop(['city_name', 'AVERAGE_PRECIPITATION','record_date','AVERAGE_HUMIDITY','AVERAGE_CLOUDINESS'], axis=1)
df_final = df_final.drop(['city_name', 'AVERAGE_PRECIPITATION','record_date','AVERAGE_HUMIDITY','AVERAGE_CLOUDINESS'], axis=1)
#Converter os valores nulos para 0
df = df.fillna(0)
df_final = df_final.fillna(0)
#Converter os valores qualitativos para quantitativos
speed_diff = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
luminosity = {'DARK': 0, 'LOW_LIGHT': 1, 'LIGHT': 2}
cloudiness = {'céu claro': 0, 'céu limpo': 0, 'nuvens dispersas': 1, 'céu pouco nublado': 1, 'algumas nuvens': 1, 'nuvens quebrados': 2, 'nuvens quebradas': 2, 'tempo nublado': 3, 'nublado': 3}
rain = {'chuvisco fraco': 1, 'chuva fraca': 1, 'chuva leve': 1,  'aguaceiros fracos': 1, 'chuvisco e chuva fraca': 1, 'chuva': 2, 'chuva moderada': 2, 'aguaceiros': 2, 'trovoada com chuva leve': 2, 'chuva de intensidade pesada': 3, 'chuva de intensidade pesado': 3, 'chuva forte': 3, 'trovoada com chuva': 3}
  
df = df.replace({'AVERAGE_SPEED_DIFF': speed_diff, 'LUMINOSITY': luminosity, 'AVERAGE_RAIN': rain})
df_final = df_final.replace({'LUMINOSITY': luminosity, 'AVERAGE_CLOUDINESS': cloudiness, 'AVERAGE_RAIN': rain})

corr_matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix,vmin=-1, vmax=1, square=True, annot=True)

x = df.drop(['AVERAGE_SPEED_DIFF'],axis=1)
y = df['AVERAGE_SPEED_DIFF'].to_frame()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)


param_grid = [
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']},
    {'scaler' : [StandardScaler()]}
]

pipe = Pipeline([('scaler' , StandardScaler()),('classifier' , RandomForestClassifier(criterion='gini',max_depth=8, max_features='log2', n_estimators=200))])


timer = datetime.datetime.now()
pipe.fit(x_train, np.ravel(y_train))
predictions = pipe.predict(x_test)
print("RFC accuracy: " + str(accuracy_score(y_test,predictions)))
print("RFC time to run: " + str((datetime.datetime.now()-timer).total_seconds()*1000))


speed_diff = {0.0: 'None', 1.0: 'Low', 2.0: 'Medium', 3.0: 'High', 4.0: 'Very_High'}
d = {'RowId': list(range(1,len(predictions)+1)), 'Speed_Diff': predictions}
result = pd.DataFrame(data=d)
result = result.replace({'Speed_Diff': speed_diff})
print(result)
result.to_csv("predictions.csv", index=False)