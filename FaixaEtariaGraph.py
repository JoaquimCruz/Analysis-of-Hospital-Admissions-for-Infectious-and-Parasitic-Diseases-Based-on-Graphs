import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.cm as cm

# ---------------------------------------------------
# 1) Carregar os dados do Excel
# ---------------------------------------------------

# Ler o arquivo Excel, assumindo a primeira coluna como nomes das cidades
df = pd.read_excel("Planilhas/FaixaEtaria.xlsx", header=0)

df.columns = df.columns.astype(str)

# Nome da primeira coluna e as faixas etárias
colunas_idade = df.columns[1:]  # Todas as colunas exceto a primeira são faixas etárias

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de faixa etária
# ---------------------------------------------------

# Criar um grafo vazio
G_faixa_etaria = nx.Graph()

# Normalizar os dados das faixas etárias
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df[colunas_idade])

# Calcular a similaridade de cosseno entre as cidades
sim_matrix = cosine_similarity(dados_normalizados)

# Adicionar arestas ao grafo para cidades com similaridade acima de um limiar
limiar = 0.75
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if sim_matrix[i, j] > limiar:
            cidade1 = df.iloc[i, 0]
            cidade2 = df.iloc[j, 0]
            G_faixa_etaria.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])

# ---------------------------------------------------
# 3) Detectar comunidades no grafo
# ---------------------------------------------------

# Usar o algoritmo de comunidades
comunidades_faixa_etaria = community.greedy_modularity_communities(G_faixa_etaria, weight='weight')

# Exibir as comunidades detectadas
print("\n=== Comunidades baseadas em similaridade de faixa etaria ===")
for i, c in enumerate(comunidades_faixa_etaria):
    print(f"Comunidade {i}: {sorted(c)}")

# Criar um dicionário {cidade: índice da comunidade}
comunidade_por_no_faixa = {}
for i, c in enumerate(comunidades_faixa_etaria):
    for node in c:
        comunidade_por_no_faixa[node] = i

# ---------------------------------------------------
# 4) Visualizar o grafo com as comunidades
# ---------------------------------------------------

# Gerar cores para as comunidades
num_comunidades_faixa = len(comunidades_faixa_etaria)
cmap_faixa = cm.colormaps.get_cmap('Set3').resampled(num_comunidades_faixa)

plt.figure(figsize=(10, 10))
pos_faixa = nx.spring_layout(G_faixa_etaria, seed=42)

# Desenhar os nós com cores por comunidade
nx.draw_networkx_nodes(
    G_faixa_etaria, pos_faixa,
    node_size=500,
    node_color=[cmap_faixa(comunidade_por_no_faixa[node]) for node in G_faixa_etaria.nodes()]
)

# Desenhar as arestas
nx.draw_networkx_edges(G_faixa_etaria, pos_faixa, width=1.5)

# Desenhar os rótulos
nx.draw_networkx_labels(G_faixa_etaria, pos_faixa, font_size=9)

plt.title("Comunidades baseadas em similaridade de faixa etária")
plt.axis('off')
plt.show()

# ---------------------------------------------------
# 5) Analisar as comunidades
# ---------------------------------------------------

# Adicionar a comunidade no DataFrame
df['Comunidade_faixa_etaria'] = df['Nome da cidade'].map(comunidade_por_no_faixa)

# Calcular as médias de internações por faixa etária para cada comunidade
medias_por_comunidade = df.groupby('Comunidade_faixa_etaria')[colunas_idade].mean()
print("\n=== Médias de internações por faixa etária em cada comunidade ===")
print(medias_por_comunidade)

# ---------------------------------------------------
# 6) Cidades presentes em cada faixa etária
# ---------------------------------------------------
# Definir um limiar para considerar uma cidade como "presente" em uma faixa etária
limiar_faixa = df[colunas_idade].mean() + df[colunas_idade].std()

print("\n=== Cidades presentes em cada faixa etária (valores acima do limiar) ===")
cidades_por_faixa = {}

for faixa in colunas_idade:
    cidades_relevantes = df[df[faixa] > limiar_faixa[faixa]]['Nome da cidade']
    cidades_por_faixa[faixa] = list(cidades_relevantes)
    print(f"Faixa: {faixa}, Cidades Relevantes: {cidades_por_faixa[faixa]}")

# Visualizar subgrafos para cada faixa etária
for faixa, cidades in cidades_por_faixa.items():
    print(f"\nProcessando Subgrafo para: {faixa}, Cidades: {cidades}")
    subgraph = G_faixa_etaria.subgraph(cidades)  # Subgrafo apenas com essas cidades

    if len(subgraph.nodes) == 0:
        print(f"Atenção: Nenhum nó encontrado no subgrafo para {faixa}")
        continue

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph, seed=42)

    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(subgraph, pos, width=1.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=9)

    plt.title(f"Grafo de Cidades Relevantes para '{faixa}'")
    plt.axis('off')
    plt.show()
