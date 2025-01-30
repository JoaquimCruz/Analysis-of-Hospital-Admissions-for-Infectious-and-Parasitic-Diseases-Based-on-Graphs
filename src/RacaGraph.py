import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import seaborn as sns
from matplotlib.cm import ScalarMappable
import random

# Configurar o backend para salvar gráficos em arquivo
import matplotlib
matplotlib.use('Agg')  # Backend para salvar gráficos

# ---------------------------------------------------
# 1) Carregar os dados do Excel (incluindo população)
# ---------------------------------------------------

# Carregar o arquivo Excel com colunas: Cidade, População, Raça Branca, Raça Negra, Raça Parda, Raça Amarela
df = pd.read_excel("/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/Raca.xlsx", header=None)
df.columns = ["Cidade", "Populacao", "Raça Branca", "Raça Negra", "Raça Parda", "Raça Amarela"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

print("\n=== DataFrame Carregado com Dados de Raças e População ===")
print(df.head())

# Capturar os nomes das colunas das raças para uso posterior
colunas_raca = ["Raça Branca", "Raça Negra", "Raça Parda", "Raça Amarela"]

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de distribuição racial
# ---------------------------------------------------

G_raca = nx.Graph()

if df[colunas_raca].isnull().any().any():
    raise ValueError("Existem valores nulos nas colunas raciais do DataFrame!")

# Normalizar os dados das raças
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df[colunas_raca])

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
            G_raca.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])

# ---------------------------------------------------
# 3) Detectar comunidades no grafo
# ---------------------------------------------------

comunidades_raca = community.greedy_modularity_communities(G_raca, weight='weight')

print("\n=== Comunidades baseadas em similaridade racial ===")
for i, c in enumerate(comunidades_raca):
    print(f"Comunidade {i}: {sorted(c)}")

comunidade_por_no_raca = {}
for i, c in enumerate(comunidades_raca):
    for node in c:
        comunidade_por_no_raca[node] = i

# ---------------------------------------------------
# 4) Salvar e visualizar o grafo com as comunidades (Versão Ampliada e com Rótulos)
# ---------------------------------------------------

sns.set_theme(style="whitegrid")

# Definir layout Kamada-Kawai com scale maior para espaçar mais os nós
pos_raca = nx.kamada_kawai_layout(G_raca, scale=5)

# Criar uma figura maior
fig, ax = plt.subplots(figsize=(20, 20))

# Preparar cores para as comunidades
comunidades_ordenadas_raca = sorted(set(comunidade_por_no_raca.values()))
num_comunidades_raca = len(comunidades_ordenadas_raca)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=num_comunidades_raca - 1)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

node_colors_raca = [comunidade_por_no_raca[node] for node in G_raca.nodes()]
cores_mapeadas_raca = [cmap(norm(comunidade)) for comunidade in node_colors_raca]

# Desenhar os nós
nx.draw_networkx_nodes(
    G_raca,
    pos_raca,
    node_size=200,             # Nó um pouco menor
    node_color=cores_mapeadas_raca,
    cmap=cmap,
    alpha=0.9,                 # Leve transparência
    ax=ax
)

# Desenhar as arestas
nx.draw_networkx_edges(
    G_raca,
    pos_raca,
    width=1.0,                 # Arestas mais finas
    alpha=0.5,                 # Arestas translúcidas
    edge_color='gray',
    ax=ax
)

# Desenhar rótulos das cidades (nós)
nx.draw_networkx_labels(
    G_raca,
    pos_raca,
    font_size=10,
    font_color='black',
    ax=ax
)

ax.set_title("Comunidades baseadas em similaridade racial (Versão Ampliada)")
ax.axis('off')

# Barra de cores para as comunidades
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Índice da Comunidade')

# Pasta de saída (já definida antes)
output_dir = "Grafos"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "grafoComunidadesRaca_grande.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafo ampliado salvo como '{output_file}'")

# ---------------------------------------------------
# 5) Analisar as comunidades e imprimir raças dominantes
# ---------------------------------------------------

df['Comunidade_raca'] = df.index.map(comunidade_por_no_raca)

medias_por_comunidade_raca = df.groupby('Comunidade_raca')[colunas_raca].mean()
racas_dominantes = medias_por_comunidade_raca.idxmax(axis=1)

print("\n=== Raças dominantes por comunidade ===")
for comunidade, raca in racas_dominantes.items():
    print(f"Comunidade {comunidade}: {raca}")

# ---------------------------------------------------
# 6) Gerar e salvar matrizes de adjacência para cada comunidade racial
# ---------------------------------------------------

pasta_matrizes_raca = "matrizes_de_adjacencia_raca"
os.makedirs(pasta_matrizes_raca, exist_ok=True)

for i, comunidade in enumerate(comunidades_raca):
    subgrafo = G_raca.subgraph(comunidade)
    nos_ordenados = sorted(subgrafo.nodes())
    matriz_adj = nx.to_pandas_adjacency(subgrafo, nodelist=nos_ordenados)
    arquivo_saida = os.path.join(pasta_matrizes_raca, f"matriz_adjacencia_comunidade_raca_{i}.txt")
    matriz_adj.to_csv(arquivo_saida, sep='\t')
    print(f"Matriz de adjacência da Comunidade {i} (raça) salva em: {arquivo_saida}")

# ---------------------------------------------------
# 7) Análise Detalhada das Comunidades Racial e Porcentagem de Internação por População
# ---------------------------------------------------

print("\n=== Análise Detalhada das Comunidades Racial ===")
for i, comunidade in enumerate(comunidades_raca):
    subgrafo = G_raca.subgraph(comunidade)
    num_nos = subgrafo.number_of_nodes()
    num_arestas = subgrafo.number_of_edges()
    densidade = nx.density(subgrafo)
    graus = dict(subgrafo.degree())
    grau_medio = sum(graus.values()) / num_nos if num_nos > 0 else 0

    print(f"\nComunidade {i}:")
    print(f" - Número de cidades: {num_nos}")
    print(f" - Número de conexões (arestas): {num_arestas}")
    print(f" - Densidade: {densidade:.4f}")
    print(f" - Grau médio: {grau_medio:.2f}")
    print(f" - Cidades: {sorted(comunidade)}")

# Cálculo da população total por comunidade
populacao_por_comunidade_raca = df.groupby('Comunidade_raca')['Populacao'].sum()

# Cálculo do total de internações por comunidade (soma de todas as raças)
internacoes_por_comunidade_raca = df.groupby('Comunidade_raca')[colunas_raca].sum().sum(axis=1)

# Cálculo da relação porcentual de internações em relação à população
relacao_porcentagem_raca = (internacoes_por_comunidade_raca / populacao_por_comunidade_raca) * 100

print("\n=== População Total por Comunidade (Raça) ===")
for comunidade, populacao in populacao_por_comunidade_raca.items():
    print(f"Comunidade {comunidade}: {populacao:,} pessoas")

print("\n=== Relação Porcentual de Internações em relação à População da Comunidade (Raça) ===")
for comunidade in relacao_porcentagem_raca.index:
    porcentagem = relacao_porcentagem_raca[comunidade]
    print(f"Comunidade {comunidade}: {porcentagem:.2f}%")

# ---------------------------------------------------
# 8) Totais Absolutos de Pessoas por Raça em Cada Comunidade e Totais Gerais
# ---------------------------------------------------

# Adicionar uma coluna com o total de internações por cidade (soma de raças)
df['Total'] = df[colunas_raca].sum(axis=1)

# Agrupar por comunidade e somar total de pessoas por raça
totais_por_comunidade_raca = df.groupby('Comunidade_raca')[colunas_raca].sum()

print("\n=== Totais Absolutos de Pessoas por Raça em Cada Comunidade ===")
for comunidade, totais in totais_por_comunidade_raca.iterrows():
    print(f"\nComunidade {comunidade}:")
    for raca, total in totais.items():
        print(f" - {raca}: {int(total):,} pessoas")

totais_gerais_raca = df[colunas_raca].sum()
print("\n=== Totais Gerais por Raça no Grafo ===")
for raca, total in totais_gerais_raca.items():
    print(f" - {raca}: {int(total):,} pessoas")


print("\n=== Percentual de Internações por Raça em Cada Comunidade ===")
for comunidade, totais in totais_por_comunidade_raca.iterrows():
    soma_comunidade = totais.sum()
    print(f"\nComunidade {comunidade}:")
    if soma_comunidade > 0:
        for raca, total in totais.items():
            percentual = (total / soma_comunidade) * 100
            print(f" - {raca}: {percentual:.2f}% de internações")
    else:
        print(" - Sem dados de internações nessa comunidade.")


# ---------------------------------------------------
# 9) Identificar os Hubs em Cada Comunidade
# ---------------------------------------------------

print("\n=== Hubs (Nodos Mais Conectados) por Comunidade ===")

# Criar um dicionário para armazenar o hub de cada comunidade
hubs_por_comunidade = {}

for i, comunidade in enumerate(comunidades_raca):
    subgrafo = G_raca.subgraph(comunidade)  # Criar subgrafo da comunidade
    grau_centralidade = nx.degree_centrality(subgrafo)  # Calcular grau de centralidade

    # Encontrar o nó com maior centralidade na comunidade
    hub = max(grau_centralidade, key=grau_centralidade.get)
    hubs_por_comunidade[i] = hub

    print(f"Comunidade {i}: Hub → {hub} (Grau de Centralidade: {grau_centralidade[hub]:.4f})")

# ---------------------------------------------------
# 10) Destacar os Hubs no Grafo
# ---------------------------------------------------

# Criar uma nova cópia do layout para garantir consistência
pos_hubs = pos_raca.copy()

fig, ax = plt.subplots(figsize=(20, 20))

# Desenhar nós normalmente
nx.draw_networkx_nodes(
    G_raca,
    pos_hubs,
    node_size=200,
    node_color=cores_mapeadas_raca,
    cmap=cmap,
    alpha=0.9,
    ax=ax
)

# Desenhar arestas
nx.draw_networkx_edges(
    G_raca,
    pos_hubs,
    width=1.0,
    alpha=0.5,
    edge_color='gray',
    ax=ax
)

# Destacar os hubs em vermelho e aumentar o tamanho deles
nx.draw_networkx_nodes(
    G_raca,
    pos_hubs,
    nodelist=hubs_por_comunidade.values(),
    node_size=500,
    node_color='red',
    edgecolors='black',
    linewidths=2,
    label="Hubs",
    ax=ax
)

# Adicionar rótulos dos nós
nx.draw_networkx_labels(
    G_raca,
    pos_hubs,
    font_size=10,
    font_color='black',
    ax=ax
)

ax.set_title("Hubs por Comunidade na Rede Racial")
ax.axis('off')

# Adicionar legenda
plt.legend(scatterpoints=1, loc='upper right')

# Salvar a imagem do grafo com hubs destacados
output_file_hubs = os.path.join(output_dir, "grafoComunidadesRaca_hubs.png")
plt.savefig(output_file_hubs, dpi=300, bbox_inches='tight')
plt.close()
print(f"Grafo com hubs destacado salvo como '{output_file_hubs}'")
