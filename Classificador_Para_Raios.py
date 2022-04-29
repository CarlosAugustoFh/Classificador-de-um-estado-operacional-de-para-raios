# -*- coding: utf-8 -*-
# UNIVERSIDADE FEDERAL DE CAMPINA GRANDE
# CENTRO DE ENGENHARIA ELÉTRICA E INFORMÁTICA
# UNIDADE ACADÊMICA DE ENGENHARIA ELÉTRICA
# INTELIGÊNCIA ARTIFICIAL E CIÊNCIA DE DADOS APLICADAS A SISTEMAS ELÉTRICOS

# Carlos Augusto Soares de Oliveira Filho - 115.111.503 

# Atividade 4.1: Classificador de para raios:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#pip install scikit-plot

import scikitplot as skplt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


# PRÉ PROCESSAMENTOS DE DADOS
amostras_bom = pd.read_csv('corrente_PR_BOM.csv', header=None)
amostras_def = pd.read_csv('corrente_PR_DEF.csv', header=None)


# Amostras antes do pré processamento:
index = np.random.randint(0, amostras_bom.values.shape[0])
plt.plot(amostras_bom.values[index])


# Pré processamento:
size_bom = amostras_bom.values.shape[1]
size_def = amostras_def.values.shape[1]


# Reduz a resolução para 250 amostras
x_bom = amostras_bom.values[::, ::(size_bom//250)]
x_bom = x_bom[::, 50::] # Descarta as 50 primeiras (200 amostras restantes)


# Reduz a resolução para 250 amostras
x_def = amostras_def.values[::, ::(size_def//250)]
x_def = x_def[::, 50::] # Descarta as 50 primeiras (200 amostras restantes)


# Amostras depois do pré processamento:
plt.plot(x_bom[index])


# PREPARANDO OS BANCOS DE TREINAMENTO E TESTE
# Criando os alvos
y_bom = np.ones(x_bom.shape[0]) # Rótulo 1 para o pararaio SEM defeito
y_def = np.zeros(x_def.shape[0]) # Rótulo 0 para o para raio COM defeito


# Juntando os dados em um único banco
x = np.concatenate((x_bom, x_def), axis=0)
y = np.concatenate((y_bom, y_def), axis=0)


# OBS: O banco ainda está mal 'misturado'. Cuidaremos disso a seguir
# Separando os bancos de teste e treinamento
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)


# O shuffle=True faz com que o banco seja embaralhado no processo de separação.
# Com isso resolvemos o problema descrito na célula anterior.

# CRIANDO E TREINANDO O CLASSIFICADOR
# Criando o classificador
net = MLPClassifier(solver = 'lbfgs', max_iter = 500, hidden_layer_sizes=(100))


# Treinando o classificador
modelo_ajustado = net.fit(x_train, y_train)


# AVALIANDO O MODELO
# Estima a precisao do modelo a partir da base de teste
score = modelo_ajustado.score(x_test, y_test)
print('Precisão:', score*100, '%')


# Calcula as previsoes do modelo a partir da base de teste
previsoes = modelo_ajustado.predict(x_test)
prevpb = modelo_ajustado.predict_proba(x_test)

precisao = accuracy_score(y_test, previsoes)
print('Acurácia:', precisao*100, '%')

print(classification_report(y_test, previsoes))


# PREVISÕES
index = np.random.randint(0, 20)

y_exemplo = y_test[index]
previsao = previsoes[index]
print('Rótulo:', 'Sem' if y_exemplo==1 else 'Com', 'defeito');
print('Previsão:', 'Sem' if previsao==1 else 'Com', 'defeito');
print('\n')

plt.plot(x_test[index])


# Plotar utilizando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix(y_test, previsoes)
plt.show()


# Plotar a ROC
skplt.metrics.plot_roc(y_test, prevpb)
plt.show()

skplt.metrics.plot_precision_recall(y_test, prevpb)
plt.show()

