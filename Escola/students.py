import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import datetime


split_value = 0.5

pd.set_option('display.max_columns',None)
#Ler ambos os ficheiros CSV
df = pd.read_csv("StudentsPerformance.csv") 

#Converter os valores qualitativos para quantitativos
gender = {'male': 0, 'female': 1}

race = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}

parental_education = {'high school': 0, 'some high school': 0, 'associate\'s degree': 1, 'some college': 2, 'bachelor\'s degree': 2, 'master\'s degree': 3}

lunch = {'free/reduced': 0, 'standard': 1}

test_preparation = {'none': 0, 'completed': 1}

df = df.replace({'gender': gender, 'race/ethnicity': race, 'parental level of education': parental_education, 'lunch': lunch, 'test preparation course': test_preparation})
results={}
results['RowId'] = list(range(1,int(len(df.index)*split_value)+1))

corr_matrix = df.corr()
f, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix,vmin=-1, vmax=1, square=True, annot=True)


param_grid = [
    {'classifier' : [Ridge()],
    'classifier__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'classifier__tol': [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000]},
]

targets = ["writing score"]

for label in targets:
    x = df.drop(label,axis=1)
    y = df[label]
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=split_value,random_state=2021)
    
    #Ridge
    timer=datetime.datetime.now()
    pipe = Pipeline([('classifier' , Ridge())])
    pipe.fit(x_train,y_train)

    predictions = pipe.predict(x_test)
    solution = y_test
    print(f"ridge regression x_test error for {label}: {str(mean_absolute_error(y_test,predictions))}")
    print(f"ridge regression time to run for {label}: {str((datetime.datetime.now()-timer).total_seconds()*1000)}")      
    
    #Decision Tree Regressor
    timer=datetime.datetime.now()
    pipe = Pipeline([('classifier' , DecisionTreeRegressor())])
    pipe.fit(x_train,y_train)
    
    predictions = pipe.predict(x_test)
    solution = y_test
    print(f"decision tree regression x_test error for {label}: {str(mean_absolute_error(y_test,predictions))}")
    print(f"decision tree regression time to run for {label}: {str((datetime.datetime.now()-timer).total_seconds()*1000)}")
    
    #Linear Regression
    timer=datetime.datetime.now()
    pipe = Pipeline([('classifier' , LinearRegression())])
    pipe.fit(x_train,y_train)

    predictions = pipe.predict(x_test)
    solution = y_test
    print(f"linear regression x_test error for {label}: {str(mean_absolute_error(y_test,predictions))}")
    print(f"linear regression time to run for {label}: {str((datetime.datetime.now()-timer).total_seconds()*1000)}")
    
    #Elastic Net
    timer=datetime.datetime.now()
    pipe = Pipeline([('classifier' , ElasticNet())])
    pipe.fit(x_train,y_train)

    predictions = pipe.predict(x_test)
    solution = y_test
    print(f"elastic net regression x_test error for {label}: {str(mean_absolute_error(y_test,predictions))}")
    print(f"elastic net time to run for {label}: {str((datetime.datetime.now()-timer).total_seconds()*1000)}")
    
    
    print(pipe.get_params().keys())
    search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv= 5,refit=True)
    timer=datetime.datetime.now()
    search.fit(x_train, np.ravel(y_train))
    
    predictions = search.predict(x_test)
    print(search.best_params_)
    print(f"ridge regression x_test error for {label}: {str(mean_absolute_error(y_test,predictions))}")
    print(f"ridge regression time to run for {label}: {str((datetime.datetime.now()-timer).total_seconds()*1000)}")
    
    results[label]=predictions
    results[label+"_real"]=solution
    
    result = pd.DataFrame(data=results)
    print(result)
    result.to_csv("school.csv", index=False)
