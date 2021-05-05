# Google colab activities - PY

Activities in google colaboratory to learn about Machine Learning.


## Libraries


- [Pandas](https://pandas.pydata.org/)
- [SKlearn](https://sklearn.org/)


## Notes

### *Modelo de treinamento - LINEAR SVC

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

<hr>

### *Ensinando o algoritimo trabalhando com matrizes

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

#### *Definindo espaçamento

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min)/ pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/ pixels)

#### *Criar grid (x mesclado com y)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

#### *Pegar o modelo e prever para os pontos

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

#### *Plotar

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)

#### *Analisar a curva de decisao - Decision Boundary (modificar a curva com o SEED)

plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

#### *Testando precisão - baseline

import numpy as np
baseline = np.ones(540)
acuracia = accuracy_score(teste_y, baseline)*100
print("A acuracia do algoritimo de baseline foi %.2f%%" % acuracia)

<hr>

### *Estimadores não lineares e vetor de suporte de maquina

#### *Modelo de treinamento - não linear

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)
raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,
                                                         stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

#### *Ensinando o algoritimo

data_x = teste_x[:,0]
data_y = teste_x[:,1]

data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(data_x, data_y, c=teste_y, s=1)
