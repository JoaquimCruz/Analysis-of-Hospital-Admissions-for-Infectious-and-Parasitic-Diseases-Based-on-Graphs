import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

# 1. Ler o arquivo Excel (sem cabecalho)
colunas = [
    'Nome da cidade', 'Populacao', 'Numero de internacoes', 'Raca branca', 'Raca preta',
    'Raca parda', 'Raca amarela', 'Internacoes de <1 ano', 
    'Internacoes de 1-4 anos', ' Internacoes de 5-14 anos', 'Internacoes de 15-24 anos', 
    'Internacoes de 25-34 anos', 'Internacoes de 35-44 anos', 'Internacoes de 45-54 anos', ' Internacoes de 55-64 anos', 
    'Internacoes d+65 anos', 'Sexo masculino', 'Sexo feminino'
]
df = pd.read_excel('Planilhas/Dados.xlsx', header=None, names=colunas)

# ---------------------------------------------------
# 1) Cria o MultiGraph
# ---------------------------------------------------
def criar_multigrafo(df):
    G = nx.MultiGraph()
    for _, row in df.iterrows():
        cidade = row['Nome da cidade']
        G.add_node(cidade)
    return G

def adicionar_arestas_por_similaridade(G, df, colunas_atributo, limiar=0.7, nome_atributo='atributo'):
    """
    Computa a similaridade de cosseno entre as cidades
    com base nas colunas_atributo e adiciona arestas
    no MultiGraph G para as cidades que ultrapassam o limiar.
    """
    # 1. Seleciona as colunas
    atributos = df[colunas_atributo]
    
    # 2. Normaliza os dados
    scaler = StandardScaler()
    atributos_normalizados = scaler.fit_transform(atributos)
    
    # 3. Calcula a similaridade de cosseno
    sim = cosine_similarity(atributos_normalizados)
    
    # 4. Adiciona arestas (somente se sim > limiar)
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] > limiar:
                cidade1 = df.iloc[i]['Nome da cidade']
                cidade2 = df.iloc[j]['Nome da cidade']
                G.add_edge(
                    cidade1,
                    cidade2,
                    key=nome_atributo,
                    weight=sim[i,j]  # ou outro critério
                )
                print(
                    f"{cidade1} conectado a {cidade2} pelo atributo '{nome_atributo}' "
                    f"(similaridade: {sim[i, j]:.2f})"
                )

# Cria um MultiGraph vazio
G_multi = criar_multigrafo(df)

# Definindo um limiar
limiar = 0.75

# Arestas baseadas em Raca
cols_raca = ['Raca branca', 'Raca preta', 'Raca parda', 'Raca amarela']
adicionar_arestas_por_similaridade(G_multi, df, cols_raca, limiar, 'raca')

# Arestas baseadas em Sexo
cols_sexo = ['Sexo masculino', 'Sexo feminino']
adicionar_arestas_por_similaridade(G_multi, df, cols_sexo, limiar, 'sexo')

# Arestas baseadas em Populacao
cols_pop = ['Populacao']
adicionar_arestas_por_similaridade(G_multi, df, cols_pop, limiar, 'populacao')

# Arestas baseadas em Faixas Etarias
cols_idade = [
    'Internacoes de <1 ano', 
    'Internacoes de 1-4 anos', 
    ' Internacoes de 5-14 anos',
    'Internacoes de 15-24 anos',
    'Internacoes de 25-34 anos',
    'Internacoes de 35-44 anos',
    'Internacoes de 45-54 anos',
    ' Internacoes de 55-64 anos',
    'Internacoes d+65 anos'
]
adicionar_arestas_por_similaridade(G_multi, df, cols_idade, limiar, 'faixa_etaria')

# ---------------------------------------------------
# 2) Unificar as arestas do MultiGraph em um Graph simples
# ---------------------------------------------------
# - Se houver mais de uma aresta (ex: 'raca', 'sexo') entre duas cidades,
#   iremos somar o peso e manter uma lista com os "motivos".
G_simple = nx.Graph()

for u, v, data in G_multi.edges(data=True):
    w = data.get('weight', 1.0)
    motivo = data.get('key', 'desconhecido')
    
    if G_simple.has_edge(u, v):
        # Se ja existir a aresta (u, v), soma o peso
        G_simple[u][v]['weight'] += w
        # E acrescenta o motivo
        G_simple[u][v]['motivos'].append(motivo)
    else:
        # Cria a aresta
        G_simple.add_edge(u, v, weight=w, motivos=[motivo])

# Mostra como as arestas ficaram unificadas
print("\n=== Arestas unificadas no Graph simples ===")
for (u, v, data) in G_simple.edges(data=True):
    print(f"{u} -- {v} | peso={data['weight']:.2f}, motivos={data['motivos']}")

# ---------------------------------------------------
# 3) Usar greedy_modularity_communities no Graph simples
# ---------------------------------------------------
comunidades = community.greedy_modularity_communities(G_simple, weight='weight')

print("\n=== Comunidades detectadas ===")
for i, c in enumerate(comunidades):
    print(f"Comunidade {i}: {sorted(c)}")

# ---------------------------------------------------
# 4) Plotar as comunidades coloridas (opcional)
# ---------------------------------------------------
# Vamos criar um dicionario {no: indice_comunidade}
comunidade_por_no = {}
for i, c in enumerate(comunidades):
    for node in c:
        comunidade_por_no[node] = i

# Podemos escolher uma paleta de cores
import matplotlib.cm as cm
import numpy as np

num_comunidades = len(comunidades)
cmap = cm.get_cmap('Set3', num_comunidades)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G_simple, seed=42)

nx.draw_networkx_nodes(
    G_simple, pos,
    node_size=500,
    node_color=[cmap(comunidade_por_no[node]) for node in G_simple.nodes()]
)
nx.draw_networkx_edges(G_simple, pos, width=1.5)
nx.draw_networkx_labels(G_simple, pos, font_size=9)

plt.title("Comunidades (Greedy Modularity) - Grafo unificado")
plt.axis('off')
plt.show()
