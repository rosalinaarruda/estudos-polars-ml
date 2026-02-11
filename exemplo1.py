import sys
print(sys.executable)

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Criando dataset com Polars
dados = pl.DataFrame({
    "tempo_cliente": [1, 2, 3, 4, 5, 6, 7, 8],
    "uso_servico": [10, 15, 20, 30, 40, 50, 60, 70],
    "cancelou": [1, 1, 1, 0, 0, 0, 0, 0]
})

# Separando X e y
X = dados.select(["tempo_cliente", "uso_servico"])
y = dados.select("cancelou")

# Convertendo para numpy (necessário para sklearn)
X_np = X.to_numpy()
y_np = y.to_numpy().ravel()

# Train/test split
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_np, y_np, test_size=0.25
)

# Modelo
modelo = LogisticRegression()
modelo.fit(X_treino, y_treino)

# Previsão
previsoes = modelo.predict(X_teste)

print("Acurácia:", accuracy_score(y_teste, previsoes))
