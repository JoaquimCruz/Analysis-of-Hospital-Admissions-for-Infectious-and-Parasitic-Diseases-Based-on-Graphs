import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import random

# Configurar o backend para salvar gráficos em arquivo
import matplotlib
matplotlib.use('Agg')  # Backend para salvar gráficos

# Criar a pasta "grafos" se não existir
output_dir = "Grafos"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------
# 1) Carregar os dados do Excel
# ---------------------------------------------------

# Carregar o arquivo Excel (adapte o caminho para o seu arquivo)
df = pd.read_excel("/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/FaixaEtaria.xlsx", header=None)

# Definir nomes das colunas com faixas etárias reais
df.columns = ["Cidade", "<1 ano", "1-4 anos", "5-14 anos", "15-24 anos", "25-34 anos", "35-44 anos", "45-54 anos",
              "55-64 anos", "65+ anos"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

# Exibir o DataFrame carregado
print("\n=== DataFrame Carregado com Faixas Etárias Reais ===")
print(df.head())

# Capturar os nomes das colunas das faixas etárias
colunas_idade = df.columns

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de faixa etária
# ---------------------------------------------------

# Criar um grafo vazio
G_faixa_etaria = nx.Graph()

# Verificar valores nulos
if df.isnull().any().any():
    raise ValueError("Existem valores nulos no DataFrame!")

# Normalizar os dados das faixas etárias
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df)

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

# Adicionar arestas ao grafo para cidades com similaridade acima de um limiar
for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
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
# 4) Salvar o grafo com as comunidades
# ---------------------------------------------------

# Gerar uma lista de cores para as comunidades
num_comunidades_faixa = len(comunidades_faixa_etaria)
cores = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_comunidades_faixa)]

plt.figure(figsize=(10, 10))
pos_faixa = nx.spring_layout(G_faixa_etaria, seed=42)

# Desenhar os nós com cores por comunidade
nx.draw_networkx_nodes(
    G_faixa_etaria, pos_faixa,
    node_size=500,
    node_color=[cores[comunidade_por_no_faixa[node]] for node in G_faixa_etaria.nodes()]
)

# Desenhar as arestas
nx.draw_networkx_edges(G_faixa_etaria, pos_faixa, width=1.5)

# Desenhar os rótulos
nx.draw_networkx_labels(G_faixa_etaria, pos_faixa, font_size=9)

# Salvar o gráfico como um arquivo PNG
output_file = os.path.join(output_dir, "grafoComunidadesFaixaEtaria.png")
plt.title("Comunidades baseadas em similaridade de faixa etária")
plt.axis('off')
plt.savefig(output_file, dpi=300)  # Salvar em alta resolução
plt.close()

print(f"Grafo salvo como '{output_file}'")

# ---------------------------------------------------
# 5) Analisar as comunidades e imprimir faixas dominantes
# ---------------------------------------------------

# Adicionar a comunidade correspondente ao DataFrame
df['Comunidade_faixa_etaria'] = df.index.map(comunidade_por_no_faixa)

# Calcular as médias de internações por faixa etária para cada comunidade
medias_por_comunidade = df.groupby('Comunidade_faixa_etaria')[colunas_idade].mean()

# Determinar a faixa etária dominante em cada comunidade
faixas_dominantes = medias_por_comunidade.idxmax(axis=1)

# Imprimir as faixas etárias dominantes no terminal
print("\n=== Faixas etárias dominantes por comunidade ===")
for comunidade, faixa in faixas_dominantes.items():
    print(f"Comunidade {comunidade}: {faixa}")

# ---------------------------------------------------
# 6) Análise Detalhada das Comunidades
# ---------------------------------------------------
print("\n=== Análise Detalhada das Comunidades ===")

# Criar subgrafos para cada comunidade e calcular métricas
for i, comunidade in enumerate(comunidades_faixa_etaria):
    # Criar o subgrafo da comunidade
    subgrafo = G_faixa_etaria.subgraph(comunidade)
    
    # Número de nós e arestas
    num_nos = subgrafo.number_of_nodes()
    num_arestas = subgrafo.number_of_edges()
    
    # Densidade do subgrafo (0 a 1, quão conectado está)
    densidade = nx.density(subgrafo)
    
    # Grau médio dos nós na comunidade
    graus = dict(subgrafo.degree())
    grau_medio = sum(graus.values()) / num_nos if num_nos > 0 else 0

    # Filtrar os dados das cidades pertencentes à comunidade
    dados_comunidade = df.loc[comunidade]
    
    # Calcular o número total de pessoas por faixa etária na comunidade
    totais_faixa_etaria = dados_comunidade[colunas_idade].sum()

    # Exibir as informações calculadas
    print(f"\nComunidade {i}:")
    print(f" - Número de cidades: {num_nos}")
    print(f" - Número de conexões (arestas): {num_arestas}")
    print(f" - Densidade: {densidade:.4f}")
    print(f" - Grau médio: {grau_medio:.2f}")
    print(f" - Cidades: {sorted(comunidade)}")
    
    print(" - Totais por faixa etária:")
    for faixa, total in totais_faixa_etaria.items():
        print(f"   - {faixa}: {int(total):,} pessoas")
