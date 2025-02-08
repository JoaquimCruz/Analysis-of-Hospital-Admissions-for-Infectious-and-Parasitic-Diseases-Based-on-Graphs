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

# Carregar o arquivo Excel com dados de faixa etária e população
df = pd.read_excel("Planilhas/FaixaEtaria.xlsx", header=None)

# Definir nomes das colunas: Cidade, Populacao, e faixas etárias
df.columns = ["Cidade", "Populacao", "<1 ano", "1-4 anos", "5-14 anos", "15-24 anos", 
              "25-34 anos", "35-44 anos", "45-54 anos", "55-64 anos", "65+ anos"]

# Configurar a primeira coluna como índice das cidades
df.set_index("Cidade", inplace=True)

print("\n=== DataFrame Carregado com Faixas Etárias e População ===")
print(df.head())

# Capturar os nomes das colunas de faixa etária (excluindo 'Populacao')
colunas_idade = ["<1 ano", "1-4 anos", "5-14 anos", "15-24 anos", 
                 "25-34 anos", "35-44 anos", "45-54 anos", "55-64 anos", "65+ anos"]

G_faixa_etaria = nx.Graph()

if df[colunas_idade].isnull().any().any():
    raise ValueError("Existem valores nulos no DataFrame!")

# Normalizar os dados das faixas etárias
scaler = StandardScaler()
dados_normalizados = scaler.fit_transform(df[colunas_idade])

# Calcular a similaridade de cosseno entre as cidades
limiar = 0.5
sim_matrix = cosine_similarity(dados_normalizados)

# Adicionar arestas ao grafo para cidades com similaridade acima do limiar
for i, cidade1 in enumerate(df.index):
    for j, cidade2 in enumerate(df.index):
        if i < j and sim_matrix[i, j] > limiar:
            G_faixa_etaria.add_edge(cidade1, cidade2, weight=sim_matrix[i, j])


comunidades_faixa_etaria = community.greedy_modularity_communities(G_faixa_etaria, weight='weight')

print("\n=== Comunidades baseadas em similaridade de faixa etária ===")
for i, c in enumerate(comunidades_faixa_etaria):
    print(f"Comunidade {i}: {sorted(c)}")

# Criar um dicionário para mapear cidade à sua comunidade
comunidade_por_no_faixa = {}
for i, c in enumerate(comunidades_faixa_etaria):
    for node in c:
        comunidade_por_no_faixa[node] = i


sns.set_theme(style="whitegrid")

# Layout ajustado para mais espaçamento
pos_faixa = nx.kamada_kawai_layout(G_faixa_etaria, scale=5)

# Criar uma figura maior
fig, ax = plt.subplots(figsize=(20, 20))

# Preparar cores para as comunidades
comunidades_ordenadas_faixa = sorted(set(comunidade_por_no_faixa.values()))
num_comunidades_faixa = len(comunidades_ordenadas_faixa)
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=0, vmax=num_comunidades_faixa - 1)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

node_colors_faixa = [comunidade_por_no_faixa[node] for node in G_faixa_etaria.nodes()]
cores_mapeadas_faixa = [cmap(norm(comunidade)) for comunidade in node_colors_faixa]

# Desenhar nós
nx.draw_networkx_nodes(
    G_faixa_etaria,
    pos_faixa,
    node_size=200,               
    node_color=cores_mapeadas_faixa,
    cmap=cmap,
    alpha=0.9,                   
    ax=ax
)

# Desenhar arestas
nx.draw_networkx_edges(
    G_faixa_etaria,
    pos_faixa,
    width=1.0,                  
    alpha=0.5,                   
    edge_color='gray',
    ax=ax
)

# Desenhar rótulos (nomes das cidades)
nx.draw_networkx_labels(
    G_faixa_etaria,
    pos_faixa,
    font_size=16,
    font_color='black',
    ax=ax
)

ax.set_title("Comunidades baseadas em similaridade de faixa etária (Figura Ampliada)")
ax.axis('off')

cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Índice da Comunidade')

output_dir = "src/Grafos"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "grafoComunidadesFaixaEtaria_grande.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()


df['Comunidade_faixa_etaria'] = df.index.map(comunidade_por_no_faixa)

medias_por_comunidade_idade = df.groupby('Comunidade_faixa_etaria')[colunas_idade].mean()
faixas_dominantes = medias_por_comunidade_idade.idxmax(axis=1)

print("\n=== Faixas etárias dominantes por comunidade ===")
for comunidade, faixa in faixas_dominantes.items():
    print(f"Comunidade {comunidade}: {faixa}")

print("\n=== Análise Detalhada das Comunidades ===")

for i, comunidade in enumerate(comunidades_faixa_etaria):
    subgrafo = G_faixa_etaria.subgraph(comunidade)
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
    
    # Calcular totais de internações por faixa etária na comunidade
    dados_comunidade = df.loc[comunidade]
    totais_faixa_etaria = dados_comunidade[colunas_idade].sum()
    print(" - Totais por faixa etária:")
    for faixa, total in totais_faixa_etaria.items():
        print(f"   - {faixa}: {int(total):,} internações")

# Cálculo da população total por comunidade
populacao_por_comunidade_idade = df.groupby('Comunidade_faixa_etaria')['Populacao'].sum()

# Cálculo do total de internações por comunidade 
internacoes_por_comunidade_idade = df.groupby('Comunidade_faixa_etaria')[colunas_idade].sum().sum(axis=1)

# Cálculo da relação porcentual de internações em relação à população
relacao_porcentagem_idade = (internacoes_por_comunidade_idade / populacao_por_comunidade_idade) * 100

print("\n=== População Total por Comunidade (Faixa Etária) ===")
for comunidade, populacao in populacao_por_comunidade_idade.items():
    print(f"Comunidade {comunidade}: {populacao:,} pessoas")

print("\n=== Relação Porcentual de Internações em relação à População da Comunidade (Faixa Etária) ===")
for comunidade in relacao_porcentagem_idade.index:
    porcentagem = relacao_porcentagem_idade[comunidade]
    print(f"Comunidade {comunidade}: {porcentagem:.2f}%")

# Cálculo do percentual de internações por faixa etária em cada comunidade
totais_por_comunidade_idade = df.groupby('Comunidade_faixa_etaria')[colunas_idade].sum()

print("\n=== Percentual de Internações por Faixa Etária em Cada Comunidade ===")
for comunidade, totais in totais_por_comunidade_idade.iterrows():
    soma_comunidade = totais.sum()
    print(f"\nComunidade {comunidade}:")
    if soma_comunidade > 0:
        for faixa, total in totais.items():
            percentual = (total / soma_comunidade) * 100
            print(f" - {faixa}: {percentual:.2f}% de internações")
    else:
        print(" - Sem dados de internações nessa comunidade.")


fig, ax = plt.subplots(figsize=(15, 8))
sns.boxplot(data=df[colunas_idade], palette="viridis", ax=ax)


ax.set_title("Distribuição das Faixas Etárias entre as Cidades")
ax.set_xlabel("Faixa Etária")
ax.set_ylabel("Número de Pessoas")
ax.grid(True, linestyle="--", alpha=0.5)


boxplot_file = os.path.join(output_dir, "boxplot_faixa_etaria.png")
plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
plt.close()
