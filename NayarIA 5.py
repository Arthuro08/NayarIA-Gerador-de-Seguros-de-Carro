import config
import pandas as pd
from sklearn import tree
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time

corfim = config.Cores.FIM 
print("Início da importação...")
dataset_bruto = pd.read_csv('seguro_carros.csv')
print("Dados importados!\n")
clf = LinearRegression()

# PIPELINE DO DATASET
dataset = pd.get_dummies(dataset_bruto, columns=['Marca', 'Nome'], dtype=int)
dados = dataset.drop(['Valor_Seguro'], axis='columns')
gabarito = dataset['Valor_Seguro']

dados_treino, dados_teste, gabarito_treino, gabarito_teste = train_test_split(dados, gabarito, test_size=0.2)

print("Hora da IA treinar!\n")
clf.fit(dados_treino, gabarito_treino)
print("IA treinada com sucesso! Hora da prova\n")
previsao = clf.predict(dados_teste)

index_teste = dataset_bruto.loc[gabarito_teste.index]

print("--- PREVISAO DA IA ---\n")

for i in range(len(dados_teste)):
    negrito = config.Cores.NEGRITO
    azul = config.Cores.AZUL
    print(f"{negrito}-Pessoa {i+1}: ganha R${index_teste.iloc[i]['Salario']} e possui {index_teste.iloc[i]['Idade']} anos")
    print(f"-Carro: {index_teste.iloc[i]['Marca']} {index_teste.iloc[i]['Nome']} {index_teste.iloc[i]['Ano']}{corfim}\n")
    print(f"Valor do seguro anual estimado: em: {azul}R${previsao[i]:.2f}{corfim}\n")

nota = clf.score(dados_teste, gabarito_teste)
if nota*100 >= 70:
    cor = config.Cores.VERDE
elif nota*100 >= 50:
    cor = config.Cores.AMARELO
else:
    cor = config.Cores.VERMELHO
print(f"\n{cor}A IA acertou {(nota * 100):.1f}% da prova!{corfim}\n")
time.sleep(3)

## AGR COM INPUT
print("Agora, é a sua vez.")
time.sleep(3)
salario = float(input("Digite seu salário: "))
idade = int(input("Digite sua idade: "))
marca = input("Digite o nome da marca: ").strip().title() # strip tira espaço invisível, title transforma em letra maiuscula primeiro e o resto minuscula
nome = input("Digite o nome do carro: ").strip().title()
ano = int(input("Digite o ano do carro: "))
valor = float(input("Digite quanto custou o carro: "))

dado_novo = {
    'Salario': [salario],
    'Idade': [idade],
    'Marca': [marca],
    'Nome': [nome],
    'Ano': [ano],
    'Valor_Carro': [valor]
}

# PIPELINE DO INPUT
dado_input = pd.DataFrame(dado_novo)
dado_input_traduzido = pd.get_dummies(dado_input, dtype=int) # Transforma o dado digitado em dummies (zeros e uns)

dado_input_preparado = dado_input_traduzido.reindex(columns=dados.columns, fill_value=0) # Alinha com todas as colunas que a IA viu no treino
# As colunas que não forem a sua marca/modelo viram 0 automaticamente, e se for um modelo q a IA nao viu ela desconsidera, não quebra

previsao_input = clf.predict(dado_input_preparado)
print(f"Valor do seguro anual estimado: em: {azul}R${previsao_input[0]:.2f}{corfim}\n")



