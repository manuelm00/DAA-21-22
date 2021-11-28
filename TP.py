import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
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

#pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows',None)
df_train = pd.read_csv("training_data.csv")
df_test = pd.read_csv("test_data.csv")
df_train = df_train.drop(['city_name', 'AVERAGE_PRECIPITATION'], axis=1)
df_test = df_test.drop(['city_name', 'AVERAGE_PRECIPITATION'], axis=1)
df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

speed_diff = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
luminosity = {'DARK': 0, 'LOW_LIGHT': 1, 'LIGHT': 2}
cloudiness = {'c�u claro': 0, 'c�u limpo': 0, 'nuvens dispersas': 1, 'c�u pouco nublado': 1, 'algumas nuvens': 1, 'nuvens quebrados': 2, 'nuvens quebradas': 2, 'tempo nublado': 3, 'nublado': 3}
rain = {'chuvisco fraco': 1, 'chuva fraca': 1, 'chuva leve': 1,  'aguaceiros fracos': 1, 'chuvisco e chuva fraca': 1, 'chuva': 2, 'chuva moderada': 2, 'aguaceiros': 2, 'trovoada com chuva leve': 2, 'chuva de intensidade pesada': 3, 'chuva de intensidade pesado': 3, 'chuva forte': 3, 'trovoada com chuva': 3}
  
df_train = df_train.replace({'AVERAGE_SPEED_DIFF': speed_diff, 'LUMINOSITY': luminosity, 'AVERAGE_CLOUDINESS': cloudiness, 'AVERAGE_RAIN': rain})
df_test = df_test.replace({'LUMINOSITY': luminosity, 'AVERAGE_CLOUDINESS': cloudiness, 'AVERAGE_RAIN': rain})

print(df_train)

x_train = df_train.drop(['AVERAGE_SPEED_DIFF','record_date'],axis=1)
y_train = df_train['AVERAGE_SPEED_DIFF'].to_frame()

x_test = df_test.drop(['record_date'],axis=1)
#y_test = df_test['AVERAGE_SPEED_DIFF'].to_frame()

clf = DecisionTreeRegressor(random_state=2021) #para numero de escolhas infinitas
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
print(list(predictions))
print(mean_absolute_error(y_test,predictions))

corr_matrix = df_train.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix,vmin=-1, vmax=1, square=True, annot=True)