import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import random
import seaborn as sns
from matplotlib.cm import ScalarMappable

# Configurar o backend para salvar gráficos em arquivo
import matplotlib
matplotlib.use('Agg')  # Backend para salvar gráficos

# Criar a pasta "grafos" se não existir
output_dir = "Grafos"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------
# 1) Carregar os dados do Excel (incluindo população)
# ---------------------------------------------------

# Carregar o arquivo Excel com as colunas: Cidade, População, Sexo Feminino, Sexo Masculino
df = pd.read_excel("/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/Sexo.xlsx", header=None)
df.columns = ["Cidade", "Populacao", "Sexo Feminino", "Sexo Masculino"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

print("\n=== DataFrame Carregado com Dados de Sexo e População ===")
print(df.head())

# Capturar os nomes das colunas de sexo para uso posterior
colunas_sexo = ["Sexo Feminino", "Sexo Masculino"]

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de distribuição de sexo
# ---------------------------------------------------

G_sexo = nx.Graph()

if df.isnull().any().any():
    raise ValueError("Existem valores nulos no DataFrame!")

# Normalizar os dados de sexo
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df[colunas_sexo])

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
            G_sexo.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])

# ---------------------------------------------------
# 3) Detectar comunidades no grafo
# ---------------------------------------------------

comunidades_sexo = community.greedy_modularity_communities(G_sexo, weight='weight')

print("\n=== Comunidades baseadas em similaridade de distribuição de sexo ===")
for i, c in enumerate(comunidades_sexo):
    print(f"Comunidade {i}: {sorted(c)}")

comunidade_por_no_sexo = {}
for i, c in enumerate(comunidades_sexo):
    for node in c:
        comunidade_por_no_sexo[node] = i

pasta_matrizes = "matrizes_de_adjacencia_sexo"

# Criar a pasta se ela não existir
os.makedirs(pasta_matrizes, exist_ok=True)


# ---- Trecho para gerar e salvar matrizes de adjacência para cada comunidade ----
for i, comunidade in enumerate(comunidades_sexo):
    # Cria o subgrafo para a comunidade atual
    subgrafo = G_sexo.subgraph(comunidade)

    # Ordenar os nós para consistência na matriz
    nos_ordenados = sorted(subgrafo.nodes())
    
    # Gerar a matriz de adjacência como DataFrame, com rótulos para linhas e colunas
    matriz_adj = nx.to_pandas_adjacency(subgrafo, nodelist=nos_ordenados)

    # Definir o caminho do arquivo para salvar a matriz de adjacência
    arquivo_saida = os.path.join(pasta_matrizes, f"matriz_adjacencia_comunidade_{i}.txt")
    
    # Salvar a matriz de adjacência em um arquivo de texto
    # Usamos sep='\t' para separar colunas por tabulação, mas pode ajustar conforme necessário
    matriz_adj.to_csv(arquivo_saida, sep='\t')
    
    print(f"Matriz de adjacência da Comunidade {i} salva em: {arquivo_saida}")
# ---- Fim do trecho de matrizes de adjacência ----

# ---------------------------------------------------
# 4) Salvar o grafo com as comunidades - Versão Melhorada
# ---------------------------------------------------

sns.set_theme(style="whitegrid")

pos_sexo = nx.kamada_kawai_layout(G_sexo)

comunidades_ordenadas = sorted(set(comunidade_por_no_sexo.values()))
num_comunidades = len(comunidades_ordenadas)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=num_comunidades - 1)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

node_colors = [comunidade_por_no_sexo[node] for node in G_sexo.nodes()]
cores_mapeadas = [cmap(norm(comunidade)) for comunidade in node_colors]

fig, ax = plt.subplots(figsize=(12, 12))
nx.draw_networkx_nodes(
    G_sexo,
    pos_sexo,
    node_size=300,
    node_color=cores_mapeadas,
    cmap=cmap,
    ax=ax
)
nx.draw_networkx_edges(
    G_sexo,
    pos_sexo,
    width=2.5,
    alpha=0.5,
    edge_color='gray',
    ax=ax
)
# Caso deseje rótulos: nx.draw_networkx_labels(G_sexo, pos_sexo, font_size=8, ax=ax)

ax.set_title("Comunidades baseadas em similaridade de distribuição de sexo")
ax.axis('off')

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Índice da Comunidade')

output_file = os.path.join(output_dir, "grafoComunidadesSexo_melhorado.png")
plt.savefig(output_file, dpi=300)
plt.close()
print(f"Grafo melhorado salvo como '{output_file}'")

# ---------------------------------------------------
# 5) Analisar as comunidades e imprimir sexo dominante
# ---------------------------------------------------

df['Comunidade_sexo'] = df.index.map(comunidade_por_no_sexo)

medias_por_comunidade = df.groupby('Comunidade_sexo')[colunas_sexo].mean()
sexo_dominante = medias_por_comunidade.idxmax(axis=1)

print("\n=== Sexo dominante por comunidade ===")
for comunidade, sexo in sexo_dominante.items():
    print(f"Comunidade {comunidade}: {sexo}")

# --- Novo trecho para calcular população total e relação porcentual de internações ---

# 1. Calcular a população total por comunidade
populacao_por_comunidade = df.groupby('Comunidade_sexo')['Populacao'].sum()

# 2. Calcular o total de internações por comunidade
internacoes_por_comunidade = df.groupby('Comunidade_sexo')[colunas_sexo].sum().sum(axis=1)

# 3. Calcular a relação porcentual de internações em relação à população de cada comunidade
relacao_porcentagem = (internacoes_por_comunidade / populacao_por_comunidade) * 100

# 4. Exibir os resultados
print("\n=== População Total por Comunidade ===")
for comunidade, populacao in populacao_por_comunidade.items():
    print(f"Comunidade {comunidade}: {populacao:,} pessoas")

print("\n=== Relação Porcentual de Internações em relação à População da Comunidade ===")
for comunidade in relacao_porcentagem.index:
    porcentagem = relacao_porcentagem[comunidade]
    print(f"Comunidade {comunidade}: {porcentagem:.2f}%")

# ---------------------------------------------------
# 6) Análise Detalhada das Comunidades
# ---------------------------------------------------

print("\n=== Análise Detalhada das Comunidades ===")
for i, comunidade in enumerate(comunidades_sexo):
    subgrafo = G_sexo.subgraph(comunidade)
    num_nos = subgrafo.number_of_nodes()
    num_arestas = subgrafo.number_of_edges()
    densidade = nx.density(subgrafo)
    graus = dict(subgrafo.degree())
    grau_medio = sum(graus.values()) / num_nos if num_nos > 0 else 0
    dados_comunidade = df.loc[list(comunidade)]
    totais_sexo = dados_comunidade[colunas_sexo].sum()

    print(f"\nComunidade {i}:")
    print(f" - Número de cidades: {num_nos}")
    print(f" - Número de conexões (arestas): {num_arestas}")
    print(f" - Densidade: {densidade:.4f}")
    print(f" - Grau médio: {grau_medio:.2f}")
    print(f" - Cidades: {sorted(comunidade)}")
    print(" - Totais por sexo:")
    for sexo, total in totais_sexo.items():
        print(f"   - {sexo}: {int(total):,} pessoas")

totais_gerais = df[colunas_sexo].sum()
print("\n=== Totais Gerais por Sexo no Grafo ===")
for sexo, total in totais_gerais.items():
    print(f" - {sexo}: {int(total):,} pessoas")

