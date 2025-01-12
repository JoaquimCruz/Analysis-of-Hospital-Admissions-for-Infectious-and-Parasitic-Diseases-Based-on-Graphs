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
df = pd.read_excel("/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/Sexo.xlsx", header=None)

# Definir nomes das colunas
df.columns = ["Cidade", "Sexo Feminino", "Sexo Masculino"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

# Exibir o DataFrame carregado
print("\n=== DataFrame Carregado com Dados de Sexo ===")
print(df.head())

# Capturar os nomes das colunas de sexo
colunas_sexo = df.columns

# ---------------------------------------------------
# 2) Criar o grafo com base em similaridade de distribuição de sexo
# ---------------------------------------------------

# Criar um grafo vazio
G_sexo = nx.Graph()

# Verificar valores nulos
if df.isnull().any().any():
    raise ValueError("Existem valores nulos no DataFrame!")

# Normalizar os dados de sexo
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df)

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

# Adicionar arestas ao grafo para cidades com similaridade acima de um limiar
for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
            G_sexo.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])

# ---------------------------------------------------
# 3) Detectar comunidades no grafo
# ---------------------------------------------------

# Usar o algoritmo de comunidades
comunidades_sexo = community.greedy_modularity_communities(G_sexo, weight='weight')

# Exibir as comunidades detectadas
print("\n=== Comunidades baseadas em similaridade de distribuição de sexo ===")
for i, c in enumerate(comunidades_sexo):
    print(f"Comunidade {i}: {sorted(c)}")

# Criar um dicionário {cidade: índice da comunidade}
comunidade_por_no_sexo = {}
for i, c in enumerate(comunidades_sexo):
    for node in c:
        comunidade_por_no_sexo[node] = i

# ---------------------------------------------------
# 4) Salvar o grafo com as comunidades
# ---------------------------------------------------

# Gerar uma lista de cores para as comunidades
num_comunidades_sexo = len(comunidades_sexo)
cores = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_comunidades_sexo)]

plt.figure(figsize=(10, 10))
pos_sexo = nx.spring_layout(G_sexo, seed=42)

# Desenhar os nós com cores por comunidade
nx.draw_networkx_nodes(
    G_sexo, pos_sexo,
    node_size=500,
    node_color=[cores[comunidade_por_no_sexo[node]] for node in G_sexo.nodes()]
)

# Desenhar as arestas
nx.draw_networkx_edges(G_sexo, pos_sexo, width=1.5)

# Desenhar os rótulos
nx.draw_networkx_labels(G_sexo, pos_sexo, font_size=9)

# Salvar o gráfico como um arquivo PNG
output_file = os.path.join(output_dir, "grafoComunidadesSexo.png")
plt.title("Comunidades baseadas em similaridade de distribuição de sexo")
plt.axis('off')
plt.savefig(output_file, dpi=300)  # Salvar em alta resolução
plt.close()

print(f"Grafo salvo como '{output_file}'")

# ---------------------------------------------------
# 5) Analisar as comunidades e imprimir sexo dominante
# ---------------------------------------------------

# Adicionar a comunidade correspondente ao DataFrame
df['Comunidade_sexo'] = df.index.map(comunidade_por_no_sexo)

# Calcular as médias de proporções de sexo para cada comunidade
medias_por_comunidade = df.groupby('Comunidade_sexo')[colunas_sexo].mean()

# Determinar o sexo dominante em cada comunidade
sexo_dominante = medias_por_comunidade.idxmax(axis=1)

# Imprimir o sexo dominante no terminal
print("\n=== Sexo dominante por comunidade ===")
for comunidade, sexo in sexo_dominante.items():
    print(f"Comunidade {comunidade}: {sexo}")

# ---------------------------------------------------
# 6) Análise Detalhada das Comunidades
# ---------------------------------------------------

print("\n=== Análise Detalhada das Comunidades ===")

# Criar subgrafos para cada comunidade e calcular métricas
for i, comunidade in enumerate(comunidades_sexo):
    # Criar o subgrafo da comunidade
    subgrafo = G_sexo.subgraph(comunidade)
    
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
    
    # Calcular o número total de pessoas por sexo na comunidade
    totais_sexo = dados_comunidade[colunas_sexo].sum()

    # Exibir as informações calculadas
    print(f"\nComunidade {i}:")
    print(f" - Número de cidades: {num_nos}")
    print(f" - Número de conexões (arestas): {num_arestas}")
    print(f" - Densidade: {densidade:.4f}")
    print(f" - Grau médio: {grau_medio:.2f}")
    print(f" - Cidades: {sorted(comunidade)}")
    
    print(" - Totais por sexo:")
    for sexo, total in totais_sexo.items():
        print(f"   - {sexo}: {int(total):,} pessoas")

# Exibir o total geral de pessoas por sexo no grafo
totais_gerais = df[colunas_sexo].sum()
print("\n=== Totais Gerais por Sexo no Grafo ===")
for sexo, total in totais_gerais.items():
    print(f" - {sexo}: {int(total):,} pessoas")
