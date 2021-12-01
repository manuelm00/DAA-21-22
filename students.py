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
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
X, y = make_multilabel_classification(n_classes=3, random_state=0)
clf = MultiOutputClassifier(KNeighborsClassifier()).fit(X, y)
clf.predict(X[-2:])

pd.set_option('display.max_columns',None)
#Ler ambos os ficheiros CSV
df = pd.read_csv("StudentsPerformance.csv") 

#Converter os valores qualitativos para quantitativos
gender = {'male': 0, 'female': 1}
#print(df['race/ethnicity'].unique())
race = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
#print(df['parental level of education'].unique())
parental_education = {'high school': 0, 'some high school': 0, 'associate\'s degree': 1, 'some college': 2, 'bachelor\'s degree': 2, 'master\'s degree': 3}
#print(df['lunch'].unique())
lunch = {'free/reduced': 0, 'standard': 1}
#print(df['test preparation course'].unique())
test_preparation = {'none': 0, 'completed': 1}

df = df.replace({'gender': gender, 'race/ethnicity': race, 'parental level of education': parental_education, 'lunch': lunch, 'test preparation course': test_preparation})

print(df)

x = df.drop(['math score'],axis=1)
y = df['math score']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=2021)


'''
lm = LinearRegression() #para numero de escolhas infinitas
lm.fit(x_train,y_train)

predictions = lm.predict(x_test)
print(len(predictions))
print("linear regression x_test error: " + str(mean_absolute_error(y_test,predictions)))'''

logmod = LogisticRegression(max_iter=2000) #para numero de escolhas infinitas
logmod.fit(x_train,np.ravel(y_train))

predictions = logmod.predict(x_test)
print("logistic regression x_test error: " + str(mean_absolute_error(y_test,predictions)))

print(predictions)

'''
corr_matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix,vmin=-1, vmax=1, square=True, annot=True)
'''