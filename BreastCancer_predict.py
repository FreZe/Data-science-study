

from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.offline

#Autor datasets: [Patricio, 2018] Patrício, M., Pereira, J., Crisóstomo, J., Matafome, P., Gomes, M., Seiça, R., & Caramelo, F. (2018). Using Resistin, glucose, age and BMI to predict the presence of breast cancer. BMC Cancer, 18(1).
#Attribute Information:

#Quantitative Attributes:
#Age (years)
#BMI (kg/m2) - body mass index
#Glucose (mg/dL)
#Insulin (µU/mL)
#HOMA
#Leptin (ng/mL)
#Adiponectin (µg/mL)
#Resistin (ng/mL)
#MCP-1(pg/dL) - monocyte chemoattractant protein-1

#Labels:
#1=Healthy controls
#2=Patients



data = pd.read_excel('dataR2.xlsx')

X = data.drop(['Classification'], axis=1)
y = data.Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

clf = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['gini', 'entropy'],
             'max_depth': range(1,30),
             'min_samples_split': range(2,10),
             'min_samples_leaf': range(1,10)}

search_params = GridSearchCV(clf, parametrs, cv=5)
search_params.fit(X_train, y_train)
best_clf = search_params.best_estimator_
best_clf.score(X_test, y_test) #выводим score, около 58%, очень низкий

parameters = {'n_estimators': [10, 20, 30], 'max_depth': [2, 5, 7, 10]}
df = RandomForestClassifier() #пробуем использовать лес
df_search = GridSearchCV(df, parameters, cv=5)
df_search.fit(X_train, y_train)
best_df = df_search.best_estimator_
best_df.score(X_test, y_test) #Леса справляются лучше. точность составила 71%







