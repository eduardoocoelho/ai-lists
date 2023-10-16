import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

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

model = GaussianNB()
model.fit(X, y)

new_data = [[mapping['Aparência']['Chuva'], mapping['Temperatura']['Fria'], mapping['Umidade']['Normal'], mapping['Ventando']['Sim']]]
odds = model.predict_proba(new_data)

not_play_odd = round(odds[0][0] * 100, 2)
play_odd = round(odds[0][1] * 100, 2)

print("Probabilidade de Não Jogar: {}%".format(not_play_odd))
print("Probabilidade de Jogar: {}%".format(play_odd))
