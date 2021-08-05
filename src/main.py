import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP

# 4 Entradas
# 3 Saídas
# Função de ativação = logística
# Taxa de aprendizado n = 0.1
# Precisão e = 10^-6
# 15 neurônios de entradas, 3 neurônios de saídas

# ALUNO:  Esaú Carvalho
# MATRÍCULA: 1610812 - 0

print(" ")
dataset = pd.read_csv('database/treinamento.csv')

# dataset.iloc[LINHA INICIAL (Inclusive) : LINHA FINAL (exclusive) , COLUNA INICIAL (inclusive) : COLUNA FINAL (exclusive)]
#Só : significa todas as linhas/colunas
X = dataset.iloc[:, 0:4].values #ENTRADAS
d = dataset.iloc[:, 4:7].values #SAÍDAS


mlp = MLP(X,d, [15, 3])

mlp.train()

print("")

print("VALIDAÇÃO DO CONJUNTO DE TESTES: ")

testdata = pd.read_csv('database/teste.csv')

T = testdata.iloc[:,0:4].values

for i in range(len(T)) :
  print(mlp.evaluate(T[i]))


# Plota gráfico dos Eqm x épocas
plt.plot(range(len(mlp.eqms)), mlp.eqms)
plt.show()