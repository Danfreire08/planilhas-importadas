import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Corrija o caminho do arquivo
caminho_arquivo = '/Users/daniel/Desktop/TCC ECEME/russia_losses_personnel.csv'

# Leia o arquivo CSV
dados_csv = pd.read_csv(caminho_arquivo)

# Usando apenas a coluna 'day' como variável independente (X)
X = dados_csv[['day']]

# Adicionando uma constante para o termo independente (intercepto)
X = sm.add_constant(X)

# Usando a coluna 'personnel' como variável dependente (y)
y = dados_csv['personnel']

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando um modelo de regressão linear
modelo = sm.OLS(y_train, X_train).fit()

# Exibindo estatísticas do modelo
print(modelo.summary())

# Fazendo previsões com o conjunto de teste
y_pred = modelo.predict(X_test)

# Visualizando a regressão linear
plt.scatter(X_test['day'], y_test, color='black')
plt.plot(X_test['day'], y_pred, color='blue', linewidth=3)
plt.title('Regressão Linear')
plt.xlabel('Dia')
plt.ylabel('Pessoal')
plt.show()

