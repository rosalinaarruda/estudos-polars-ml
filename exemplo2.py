import sys
print(sys.executable)


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import polars as pl

iris = load_iris()

# Convertendo para Polars
X = pl.DataFrame(iris.data, schema=iris.feature_names)
y = pl.Series("target", iris.target)

# Convertendo para numpy
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_np, y_np, test_size=0.3
)

modelo = RandomForestClassifier()
modelo.fit(X_treino, y_treino)

previsao = modelo.predict(X_teste)

print(previsao[:5])
