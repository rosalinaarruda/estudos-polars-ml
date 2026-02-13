#ver valores nulos
print(dados.null_count())

#filtrar dados
dados_filtrados = dados.filter(pl.col("tempo_cliente") > 3)

#criar colunas
dados = dados.with_columns(
    (pl.col("uso_servico") / pl.col("tempo_cliente")).alias("uso_medio")
)


#agrupar
dados.group_by("cancelou").agg(
    pl.col("uso_servico").mean()
)

#lazy otimiza antes de executar
df.lazy().filter(...).collect()
