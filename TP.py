import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import sklearn

#Ler ambos os ficheiros CSV
df = pd.read_csv("training_data.csv", encoding = 'latin-1')
df_final = pd.read_csv("test_data.csv", encoding = 'latin-1')
#Remover colunas sem informação relevante
df = df.drop(['city_name', 'AVERAGE_PRECIPITATION'], axis=1)
df_final = df_final.drop(['city_name', 'AVERAGE_PRECIPITATION'], axis=1)
#Converter os valores nulos para 0
df = df.fillna(0)
df_final = df_final.fillna(0)
#Converter os valores qualitativos para quantitativos
speed_diff = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
luminosity = {'DARK': 0, 'LOW_LIGHT': 1, 'LIGHT': 2}
cloudiness = {'céu claro': 0, 'céu limpo': 0, 'nuvens dispersas': 1, 'céu pouco nublado': 1, 'algumas nuvens': 1, 'nuvens quebrados': 2, 'nuvens quebradas': 2, 'tempo nublado': 3, 'nublado': 3}
rain = {'chuvisco fraco': 1, 'chuva fraca': 1, 'chuva leve': 1,  'aguaceiros fracos': 1, 'chuvisco e chuva fraca': 1, 'chuva': 2, 'chuva moderada': 2, 'aguaceiros': 2, 'trovoada com chuva leve': 2, 'chuva de intensidade pesada': 3, 'chuva de intensidade pesado': 3, 'chuva forte': 3, 'trovoada com chuva': 3}
  
df = df.replace({'AVERAGE_SPEED_DIFF': speed_diff, 'LUMINOSITY': luminosity, 'AVERAGE_CLOUDINESS': cloudiness, 'AVERAGE_RAIN': rain})
df_final = df_final.replace({'LUMINOSITY': luminosity, 'AVERAGE_CLOUDINESS': cloudiness, 'AVERAGE_RAIN': rain})

x = df.drop(['AVERAGE_SPEED_DIFF'],axis=1)
y = df['AVERAGE_SPEED_DIFF'].to_frame()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)

x_train = x_train.drop(['record_date'],axis=1)
x_test = x_test.drop(['record_date'],axis=1)

clf = DecisionTreeRegressor(random_state=2) #para numero de escolhas infinitas
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
print(len(predictions))
print("decision tree x_test error: " + str(mean_absolute_error(y_test,predictions)))
#print(df_final)
#predictions = clf.predict(df_final.drop(['record_date'],axis=1))
#print(list(predictions))

lm = LinearRegression() #para numero de escolhas infinitas
lm.fit(x_train,y_train)

predictions = lm.predict(x_test)
print(len(predictions))
print("linear regression x_test error: " + str(mean_absolute_error(y_test,predictions)))

logmod = LogisticRegression(max_iter=2000) #para numero de escolhas infinitas
logmod.fit(x_train,np.ravel(y_train))

predictions = logmod.predict(x_test)
print("logistic regression x_test error: " + str(mean_absolute_error(y_test,predictions)))


speed_diff = {0.0: 'None', 1.0: 'Low', 2.0: 'Medium', 3.0: 'High', 4.0: 'Very_High'}
d = {'RowId': list(range(1,len(predictions)+1)), 'Speed_Diff': predictions}
result = pd.DataFrame(data=d)
result = result.replace({'Speed_Diff': speed_diff})
print(result)
result.to_csv("respes.csv", index=False)

'''
corr_matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix,vmin=-1, vmax=1, square=True, annot=True)
'''