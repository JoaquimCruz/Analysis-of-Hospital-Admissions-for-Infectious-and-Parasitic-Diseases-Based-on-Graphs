# Graph-Based Analysis of Hospital Admissions in Western Minas Gerais
<div style="display: inline-block;">
<img align="center" height="20px" width="60px" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/> 
<img align="center" height="20px" width="80px" src="https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg"/> 
</a> 
</div>

## Introdução

Este repositório contém um artigo que analisa o perfil das internações hospitalares na Macrorregião Oeste de Minas Gerais. O Brasil atravessa uma transição demográfica e epidemiológica complexa, caracterizada pelo aumento das doenças crônicas não transmissíveis, ao lado da persistência de doenças infecciosas e parasitárias. 

Neste estudo, utilizamos dados do Sistema de Informação Hospitalar do SUS (SIH/SUS) e aplicamos a teoria dos grafos para detectar padrões de similaridade entre os municípios.

Agrupamos as cidades com base em características como gênero, raça/etnia e faixa etária, identificando perfis predominantes de internação e outliers relevantes. Os resultados revelam uma distribuição heterogênea das internações, evidenciando a importância de políticas públicas direcionadas para reduzir desigualdades regionais na saúde.

## Artigo Científico

O artigo científico resultante desta pesquisa está disponível no repositório. Nele contém todas as informações sobre a pesquisa e sobre os resultados obtidos. [Clique aqui](https://github.com/JoaquimCruz/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/blob/main/Análise_de_internações_Hospitalares_baseadas_em_Grafos.pdf) para visualizar o documento completo.

## Tecnologias Utilizadas

O projeto foi desenvolvido em Python, utilizando as seguintes bibliotecas:

```python
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
```

## Instalação das Dependências

Para rodar o código usando Python, é necessário instalar as bibliotecas utilizadas, utilizando o comando pip install. 
### Linux
```sh
pip install pandas scikit-learn networkx matplotlib seaborn
```

### Windows
```sh
pip install pandas scikit-learn networkx matplotlib seaborn
```

### macOS
```sh
pip install pandas scikit-learn networkx matplotlib seaborn
```


