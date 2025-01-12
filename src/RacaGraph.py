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
df = pd.read_excel("/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/Raca.xlsx", header=None)

# Definir nomes das colunas
df.columns = ["Cidade", "Raça Branca", "Raça Negra", "Raça Parda", "Raça Amarela"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

# Exibir o DataFrame carregado
print("\n=== DataFrame Carregado com Dados de Raças ===")
print(df.head())

# Capturar os nomes das colunas das raças
colunas_raca = df.columns

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de distribuição racial
# ---------------------------------------------------

# Criar um grafo vazio
G_raca = nx.Graph()

# Verificar valores nulos
if df.isnull().any().any():
    raise ValueError("Existem valores nulos no DataFrame!")

# Normalizar os dados das raças
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df)

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

# Adicionar arestas ao grafo para cidades com similaridade acima de um limiar
for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
            G_raca.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])

# ---------------------------------------------------
# 3) Detectar comunidades no grafo
# ---------------------------------------------------

# Usar o algoritmo de comunidades
comunidades_raca = community.greedy_modularity_communities(G_raca, weight='weight')

# Exibir as comunidades detectadas
print("\n=== Comunidades baseadas em similaridade racial ===")
for i, c in enumerate(comunidades_raca):
    print(f"Comunidade {i}: {sorted(c)}")

# Criar um dicionário {cidade: índice da comunidade}
comunidade_por_no_raca = {}
for i, c in enumerate(comunidades_raca):
    for node in c:
        comunidade_por_no_raca[node] = i

# ---------------------------------------------------
# 4) Salvar o grafo com as comunidades
# ---------------------------------------------------

# Gerar uma lista de cores para as comunidades
num_comunidades_raca = len(comunidades_raca)
cores = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_comunidades_raca)]

plt.figure(figsize=(10, 10))
pos_raca = nx.spring_layout(G_raca, seed=42)

# Desenhar os nós com cores por comunidade
nx.draw_networkx_nodes(
    G_raca, pos_raca,
    node_size=500,
    node_color=[cores[comunidade_por_no_raca[node]] for node in G_raca.nodes()]
)

# Desenhar as arestas
nx.draw_networkx_edges(G_raca, pos_raca, width=1.5)

# Desenhar os rótulos
nx.draw_networkx_labels(G_raca, pos_raca, font_size=9)

# Salvar o gráfico como um arquivo PNG
output_file = os.path.join(output_dir, "grafoComunidadesRaca.png")
plt.title("Comunidades baseadas em similaridade racial")
plt.axis('off')
plt.savefig(output_file, dpi=300)  # Salvar em alta resolução
plt.close()

print(f"Grafo salvo como '{output_file}'")

# ---------------------------------------------------
# 5) Analisar as comunidades e imprimir raças dominantes
# ---------------------------------------------------

# Adicionar a comunidade correspondente ao DataFrame
df['Comunidade_raca'] = df.index.map(comunidade_por_no_raca)

# Calcular as médias de proporções raciais para cada comunidade
medias_por_comunidade = df.groupby('Comunidade_raca')[colunas_raca].mean()

# Determinar a raça dominante em cada comunidade
racas_dominantes = medias_por_comunidade.idxmax(axis=1)

# Imprimir as raças dominantes no terminal
print("\n=== Raças dominantes por comunidade ===")
for comunidade, raca in racas_dominantes.items():
    print(f"Comunidade {comunidade}: {raca}")

# ---------------------------------------------------
# 6) Análise Detalhada das Comunidades
# ---------------------------------------------------

print("\n=== Análise Detalhada das Comunidades ===")

# Criar subgrafos para cada comunidade e calcular métricas
for i, comunidade in enumerate(comunidades_raca):
    # Criar o subgrafo da comunidade
    subgrafo = G_raca.subgraph(comunidade)
    
    # Número de nós e arestas
    num_nos = subgrafo.number_of_nodes()
    num_arestas = subgrafo.number_of_edges()
    
    # Densidade do subgrafo (0 a 1, quão conectado está)
    densidade = nx.density(subgrafo)
    
    # Grau médio dos nós na comunidade
    graus = dict(subgrafo.degree())
    grau_medio = sum(graus.values()) / num_nos if num_nos > 0 else 0

    # Exibir as informações calculadas
    print(f"\nComunidade {i}:")
    print(f" - Número de cidades: {num_nos}")
    print(f" - Número de conexões (arestas): {num_arestas}")
    print(f" - Densidade: {densidade:.4f}")
    print(f" - Grau médio: {grau_medio:.2f}")
    print(f" - Cidades: {sorted(comunidade)}")

# 7) Análise Detalhada de Raças por Comunidade
# ---------------------------------------------------

# Adicionar uma coluna com o total de pessoas por cidade
df['Total'] = df[colunas_raca].sum(axis=1)

# Agrupar dados por comunidade e calcular o total absoluto de pessoas por raça
totais_por_comunidade = df.groupby('Comunidade_raca')[colunas_raca].sum()

# Exibir totais absolutos por raça para cada comunidade
print("\n=== Totais Absolutos de Pessoas por Raça em Cada Comunidade ===")
for comunidade, totais in totais_por_comunidade.iterrows():
    print(f"\nComunidade {comunidade}:")
    for raca, total in totais.items():
        print(f" - {raca}: {int(total):,} pessoas")  # Exibe o valor formatado com vírgulas

# Exibir o total geral de pessoas por raça no grafo
totais_gerais = df[colunas_raca].sum()
print("\n=== Totais Gerais por Raça no Grafo ===")
for raca, total in totais_gerais.items():
    print(f" - {raca}: {int(total):,} pessoas")
