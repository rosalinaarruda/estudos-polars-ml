#exemplo 1 para estudo em python explorando o dataset criado 
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#criando o dataset
np.random.seed(42)

n = 500

dataset = pl.DataFrame({
    "tempo_assinatura": np.random.randint(1, 36, n),
    "horas_assistidas_mes": np.random.randint(5, 120, n),
    "valor_mensal": np.random.uniform(20, 100, n),
    "qtd_reclamacoes": np.random.randint(0, 5, n)
})

#Quantas linhas e colunas existem no dataset
print(dataset.shape)

#mostrar os 5 primeiros registros
print(dataset.head())

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

#criar uma variável derivada, chamada uso_baixo:

dataset = dataset.with_columns(
    (
        pl.col("horas_assistidas_mes") < 30
    ).cast(pl.Int8).alias("uso_baixo")
)
print(dataset["uso_baixo"].value_counts())

#criação do target
dataset = dataset.with_columns(
    (
        (pl.col("tempo_assinatura") < 6) |
        (pl.col("horas_assistidas_mes") < 20) |
        (pl.col("qtd_reclamacoes") > 2)
    ).cast(pl.Int8).alias("cancelou")
)

#Quantos clientes cancelaram e quantos não cancelaram
print(dataset["cancelou"].value_counts())

#percentual de cancelamento para clientes novos e clientes antigos
print("Percentual de cancelamento cliente novo:",dataset.filter(pl.col("cliente_novo")==1).select(pl.col("cancelou").mean()*100).item())
print("Percentual de cancelamento cliente antigo:",dataset.filter(pl.col("cliente_novo")==0).select(pl.col("cancelou").mean()*100).item())


#treinar o modelo
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

#matriz confusao
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(matriz)

#calculo acuracia
print("Acurácia:", accuracy_score(y_test, y_pred))

# Calculo Precisao X Recall. Precisão → Quando prevê cancelamento, ele acerta? Recall → Quantos cancelamentos reais ele encontrou?
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

#calculo roc-auc: Avalia a capacidade geral do modelo separar clientes que cancelam dos que não cancelam.
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

#plotando a curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC")
plt.show()










