#код для предсказания и подсчета "собачек"
from sklearn import tree
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


data_train = pd.read_csv('dogs_n_cats.csv')
data_test = pd.read_json('dataset_209691_15 (1).txt')


clf = tree.DecisionTreeClassifier(criterion='entropy')


X_train = data_train.drop(['Вид'], axis=1)
y_train = data_train.Вид
X_test = data_test[['Длина', 'Высота', 'Шерстист', 'Гавкает', 'Лазает по деревьям']]


clf.fit(X_train, y_train)



hh = clf.predict(X_test)
l = list(hh)
l.count('собачка')






