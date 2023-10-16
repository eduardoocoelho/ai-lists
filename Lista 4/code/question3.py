import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Random Forest

data = pd.read_csv('data.csv')

mapping = {
    'Aparência': {'Sol': 0, 'Nublado': 1, 'Chuva': 2},
    'Temperatura': {'Quente': 0, 'Agradável': 1, 'Fria': 2},
    'Umidade': {'Alta': 0, 'Normal': 1},
    'Ventando': {'Sim': 0, 'Não': 1},
    'Jogar': {'Não': 0, 'Sim': 1}
}

data = pd.read_csv('data.csv').replace(mapping)

X = data[['Aparência', 'Temperatura', 'Umidade', 'Ventando']]
y = data['Jogar']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier(random_state=42)

model_rf.fit(X_train, y_train)

predictions_rf = model_rf.predict(X_test)

precision_rf = accuracy_score(y_test, predictions_rf)
print("Precisão do Random Forest:", precision_rf)


#------------------------------------------------------------------------------------------#
#Ajustar Hiperparâmetros

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model_rf_grid = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model_rf_grid, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Melhores hiperparâmetros encontrados:", best_params)

model_rf_optimized = grid_search.best_estimator_
predictions_rf_optimized = model_rf_optimized.predict(X_test)
precision_rf_optimized = accuracy_score(y_test, predictions_rf_optimized)
print("Precisão do Random Forest otimizado:", precision_rf_optimized)


#------------------------------------------------------------------------------------------#
#Ajustar Hiperparâmetros com RandomSearch

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None] + list(np.arange(10, 110, 10)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 5)
}

model_rf_random = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(estimator=model_rf_random, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
print("Melhores hiperparâmetros encontrados (RandomSearch):", best_params_random)

model_rf_random_optimized = random_search.best_estimator_
predictions_rf_random_optimized = model_rf_random_optimized.predict(X_test)
precision_rf_random_optimized = accuracy_score(y_test, predictions_rf_random_optimized)
print("Precisão do Random Forest otimizado (RandomSearch):", precision_rf_random_optimized)
