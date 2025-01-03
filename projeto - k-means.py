"""
    Neste código, exploraremos o processo K-means de aprendizado não supervisionado, cujo objetivo final é encontrar o "k" ideal e rotular o dataframe original -- base de valores de vendas de uma empresa fictícia.
"""

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Carregando o arquivo de dados e eliminando valores nulos
df = pd.read_excel("VENDAS.xlsx")
df_notnull = df.dropna()

# Selecionando a coluna com os dados de vendas
df_coluna_base = df_notnull[["VALOR_VENDA"]]

print(df_coluna_base.head())

# Normalizando os dados
scaler = StandardScaler()
df_normalizado = scaler.fit_transform(df_coluna_base)

"""
    Depois de toda as transformações e pré-processamentos iniciais, agora a parte de encontrar o "k" ideal para a base de dados.
"""

# Encontrando o número ideal de clusters usando o método do cotovelo e índice de silhueta
sse = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_normalizado)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_normalizado, kmeans.labels_))

# Visualizando o Método do Cotovelo
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), sse, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('SSE')
plt.grid()
plt.show()

# Visualizando o Índice de Silhueta
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title('Coeficiente de Silhueta')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Média do Índice de Silhueta')
plt.grid()
plt.show()

# Escolher o melhor "k" e rodar o K-means final
# Para a solução, os resultados se baseiam no Índice de Silhueta
k_valores = range(2, 11)
melhor_k = k_valores[np.argmax(silhouette_scores)]
print(f"O melhor 'k' do modelo é: {melhor_k}")

kmeans_final = KMeans(n_clusters=melhor_k, random_state=42).fit(df_normalizado)

# Adicionando os rótulos aos dados originais
df_notnull['Cluster'] = kmeans_final.labels_
print(df_notnull.head())