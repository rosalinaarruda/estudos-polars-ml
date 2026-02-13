#exemplo 1 para estudo em python explorando o dataset criado
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


np.random.seed(42)

n = 500

dataset = pl.DataFrame({
    "tempo_assinatura": np.random.randint(1, 36, n),
    "horas_assistidas_mes": np.random.randint(5, 120, n),
    "valor_mensal": np.random.uniform(20, 100, n),
    "qtd_reclamacoes": np.random.randint(0, 5, n)
})

dataset = dataset.with_columns(
    (
        (pl.col("tempo_assinatura") < 6) |
        (pl.col("horas_assistidas_mes") < 20) |
        (pl.col("qtd_reclamacoes") > 2)
    ).cast(pl.Int8).alias("cancelou")
)

X = dataset.drop("cancelou").to_pandas()
y = dataset["cancelou"].to_pandas()

#separação de treino e teste para evitar que o modelo memorize os dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#simular modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

#previsoes
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

#Quantas linhas e colunas existem no dataset
print(dataset.shape)

#mostrar os 5 primeiros registros
print(dataset.head(n=5))

#Quantos clientes cancelaram e quantos não cancelaram
print(dataset["cancelou"].value_counts())

#Média das variáveis: tempo_assinatura, horas_assistidas_mes, valor_mensal, qtd_reclamacoes
print("Média de tempo de assinatura:", dataset["tempo_assinatura"].mean())
print("Média de horas assistidas por mes:", dataset["horas_assistidas_mes"].mean())
print("Média de valor pago mensal:", dataset["valor_mensal"].mean())
print("Média de quantidade de reclamações:", dataset["qtd_reclamacoes"].mean())

#criar uma nova coluna chamada cliente_novo

dataset = dataset.with_columns(
    (
        pl.col("tempo_assinatura") < 6 
    ).cast(pl.Int8).alias("cliente_novo")
)
print(dataset["cliente_novo"].value_counts())

#criar uma variável derivada, chamado uso_baixo:

dataset = dataset.with_columns(
    (
        pl.col("horas_assistidas_mes") < 30
    ).cast(pl.Int8).alias("uso_baixo")
)
print(dataset["uso_baixo"].value_counts())

#percentual de cancelamento para clientes novos e clientes antigos
print("Percentual de cancelamento cliente novo:",dataset.filter(pl.col("cliente_novo")==1).select(pl.col("cancelou").mean()*100).item())
print("Percentual de cancelamento cliente antigo:",dataset.filter(pl.col("cliente_novo")==0).select(pl.col("cancelou").mean()*100).item())

