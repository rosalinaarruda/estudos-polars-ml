import polars as pl
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Criando dataset fictício
dados = pl.DataFrame({
    "duracao": [90, 120, 150, 110, 95],
    "explosoes": [0, 5, 8, 2, 0],
    "piadas": [10, 2, 1, 5, 8],
    "romance": [1, 1, 0, 1, 0]
})

# Labels multilabel
labels = pl.DataFrame({
    "acao": [0, 1, 1, 0, 0],
    "comedia": [1, 0, 0, 1, 1],
    "romance": [1, 1, 0, 1, 0]
})

X = dados.to_numpy()
y = labels.to_numpy()

modelo = MultiOutputClassifier(RandomForestClassifier())

modelo.fit(X, y)

novo_filme = [[100, 4, 6, 1]]
previsao = modelo.predict(novo_filme)

print(previsao)
