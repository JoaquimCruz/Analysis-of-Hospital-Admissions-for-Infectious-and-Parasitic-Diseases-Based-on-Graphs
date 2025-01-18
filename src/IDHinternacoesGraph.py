import pandas as pd

# Defina o caminho para o seu arquivo Excel
arquivo_excel = "/home/joaquim/Documents/Graph-Based-Analysis-of-Hospital-Admissions-in-Western-Minas-Gerais/Planilhas/IDH.xlsx"  # Substitua pelo caminho correto do arquivo

# Definir nomes das colunas esperadas
nomes_colunas = ["Cidade", "População", "Frequência_Total_de_Internações"]

# Ler o arquivo Excel
df = pd.read_excel(arquivo_excel, header=None, names=nomes_colunas)

# Converter colunas numéricas e remover linhas com valores faltantes
df["População"] = pd.to_numeric(df["População"], errors='coerce')
df["Frequência_Total_de_Internações"] = pd.to_numeric(df["Frequência_Total_de_Internações"], errors='coerce')
df = df.dropna(subset=["Cidade", "População", "Frequência_Total_de_Internações"])

# Calcular a taxa de internação por 100 habitantes
df["Taxa_por_100"] = (df["Frequência_Total_de_Internações"] / df["População"]) * 100

# Imprimir os resultados com o nome da cidade
print("Taxa de Internação por 100 Habitantes por Cidade:")
for _, row in df.iterrows():
    print(f"{row['Cidade']}: {row['Taxa_por_100']:.2f} internações/100 habitantes")
