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
df = pd.read_csv("training_data.csv")
print(df.drop(['AVERAGE_SPEED_DIFF','LUMINOSITY','AVERAGE_PRECIPITATION','AVERAGE_RAIN'], axis=1))


#print(df)
#print(df.head()) #shows things
#print(df.info()) #mostra tipo 4 nulos
#print(df.describe()) #does medias

#print(sns.histplot(df['AVERAGE_WIND_SPEED'], kde=True))
#print("Skewness : %f" % df['AVERAGE_WIND_SPEED'].skew())
#print("Kurtosis : %f" % df['AVERAGE_WIND_SPEED'].kurt())
#ax = sns.boxplot(x=df["AVERAGE_WIND_SPEED"])
_ = df.plot.scatter(x='AVERAGE_ATMOSP_PRESSURE', y='AVERAGE_HUMIDITY')

#O que é df_prepared???
#df_prepared['AVERAGE_WIND_SPEED'].fillna(method ='bfill')

#df.loc[~df['AVERAGE_CLOUDINESS'].isin(['céu limpo'])] #localiza

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#df[['AVG_TIME_DIFF_NORMALIZED']] = min_max_scaler.fit_transform(df[['AVERAGE_TIME_DIFF']])
#print(df['AVG_TIME_DIFF_NORMALIZED'].describe())

#standard_scaler = preprocessing.StandardScaler().fit(df[['AVERAGE_TIME_DIFF']])
#df[['AVG_TIME_DIFF_SCALED']] = standard_scaler.transform(df[['AVERAGE_TIME_DIFF']])
#print(df['AVG_TIME_DIFF_SCALED'].describe())

#est = preprocessing.KBinsDiscretizer(n_bins=4,encode='ordinal', strategy='quantile')
#df['Avg_Time_Diff_Binned'] = est.fit_transform(df[['AVERAGE_TIME_DIFF']])
#print(df.groupby(by=['Avg_Time_Diff_Binned']).count())
'''
x = df.drop(['AVERAGE_SPEED_DIFF'],axis=1)
y = df['AVERAGE_SPEED_DIFF'].to_frame()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)

x_train = x_train.drop(['city_name','record_date','LUMINOSITY','AVERAGE_CLOUDINESS','AVERAGE_PRECIPITATION','AVERAGE_RAIN'],axis=1)
x_test = x_test.drop(['city_name','record_date','LUMINOSITY','AVERAGE_CLOUDINESS','AVERAGE_PRECIPITATION','AVERAGE_RAIN'],axis=1)


clf = DecisionTreeClassifier(random_state=2021) #para numero de escolhas finitas
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
print(predictions)

#nothing really works here
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
print(precision_score(y_test,predictions,average=None))
print(recall_score(y_test,predictions,average=None))
print(roc_auc_score(y_test,predictions,average=None)) não funciona

#este não funciona pq não há nenhum exemplo de decisiontreeclassifier com números no exercicio
fpr, tpr, _ = roc_curve(y_test,predictions, pos_label='your_label')
plt.clf()
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

#acho que estes 2 nao funcionam either
f1_score(y_test,predictions,average=None)
fbeta_score(y_test,predictions,beta=0.5,average=None)

x = df.drop(['AVERAGE_TIME_DIFF'],axis=1)
y = df['AVERAGE_TIME_DIFF'].to_frame()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)

x_train = x_train.drop(['city_name','record_date','AVERAGE_SPEED_DIFF','LUMINOSITY','AVERAGE_CLOUDINESS','AVERAGE_PRECIPITATION','AVERAGE_RAIN'],axis=1)
x_test = x_test.drop(['city_name','record_date','AVERAGE_SPEED_DIFF','LUMINOSITY','AVERAGE_CLOUDINESS','AVERAGE_PRECIPITATION','AVERAGE_RAIN'],axis=1)

clf = DecisionTreeRegressor(random_state=2021) #para numero de escolhas infinitas
clf.fit(x_train,y_train)

predictions = clf.predict(x_test)
print(predictions)

print(mean_absolute_error(y_test,predictions))

print(mean_squared_error(y_test,predictions,squared=True))
print(mean_squared_error(y_test,predictions,squared=False))

#estes embaixo não funcionam ehehe
x = df.drop(['AVERAGE_TIME_DIFF'],axis=1)
y = df['AVERAGE_TIME_DIFF'].to_frame()

print("USING A DECISION TREE WITH cross_val_score (MEAN ACCURACY)...")
x = x.drop(['city_name','record_date','AVERAGE_SPEED_DIFF','LUMINOSITY','AVERAGE_CLOUDINESS','AVERAGE_PRECIPITATION','AVERAGE_RAIN'],axis=1)
clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 2, random_state=2021)
scores = cross_val_score(clf,x,y,cv=2)
print(scores)
print("RESULT: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))'''