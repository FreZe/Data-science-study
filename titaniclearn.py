#тренировка по предсказанию выживших на титанике


from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from IPython.display import SVG
from graphviz import Source
from IPython.display import display


from IPython.display import HTML # Для визуализации глубины классификации данных Решающих деревьев.
style = "<style>svg{width: 50% !important; height: 50% !important;} </style>"
HTML(style)


titanic_data = pd.read_csv('train.csv') #cчитываем данные из csv файла



titanic_data.head() #знакомимся с представлением данных.


titanic_data.isnull().sum() #колличество n/a в сериях


X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1) #записываю переменные для обучения
y = titanic_data.Survived


X.head() #проверяю представление данных в переменной X

X = pd.get_dummies(X) #разделяю строки в переменной
X = X.fillna({'Age': X.Age.median()}) #заменяю n/a на медиану возраста

clf = tree.DecisionTreeClassifier(criterion='entropy') #создаю экземпляр класса решающего дерева
clf.fit(X, y) #обучаем модель.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
X_train.head()

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

scores_data = pd.DataFrame()
max_depth_values = range(1, 100)

for max_depth in max_depth_values: # перебор глубины дерева для определения оптимальной.
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth': [max_depth], 'train_score': [train_score], 'test_score': [test_score],
                                   'cross_val_score': [mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)

scores_data.head()
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], value_vars=['train_score', 'test_score', 'cross_val_score'],
                          var_name='set_type', value_name='score')
scores_data_long.head()
sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)

best_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
cross_val_score(clf, X_test, y_test, cv=5).mean()

from sklearn.model_selection import GridSearchCV #пробуем автоматизировать подбор оптимальных параметров модели

clf = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,30)}
grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5)

grid_search_cv_clf.fit(X_train, y_train)
grid_search_cv_clf.best_params_ #определяем лучшие параметры для модели из представленных в словаре parametrs
best_clf = grid_search_cv_clf.best_estimator_ #фиксируем оптимальные параметры в классификаторе

best_clf.score(X_test, y_test) #Смотрим точность с подобраными параметрами

from sklearn.metrics import precision_score, recall_score

y_pred = best_clf.predict(X_test)
precision_score(y_test, y_pred) #cмотрим на метрику precision
recall_score(y_test, y_pred) #cмотрим на метрику recall


