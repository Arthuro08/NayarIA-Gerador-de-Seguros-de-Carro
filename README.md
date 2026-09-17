# NayarIA v0.5 - Previsão de Seguro Automotivo 🚗💨

Modelo de Machine Learning Supervisionado utilizando **Regressão Linear** treinado com um dataset em larga escala de **100.000 registros**, cobrindo **27 marcas** e **204 modelos de carros**, além de permitir que o usuário digite seus próprios dados para receber uma cotação em tempo real no terminal.

---

## 📊 Sobre o Projeto e o Dataset

O objetivo da **NayarIA v0.5** é estimar o valor anual do seguro automotivo com base em características socioeconômicas do condutor e especificações do veículo.

### Variáveis utilizadas:
* **`Salario`** (float): Renda mensal do segurado.
* **`Idade`** (int): Idade do condutor (18 a 78 anos).
* **`Marca`** (string): Montadora do veículo (27 marcas: populares, SUVs, elétricos e luxo).
* **`Nome`** (string): Modelo específico do carro (204 modelos catalogados).
* **`Ano`** (int): Ano de fabricação do veículo (2005 a 2025).
* **`Valor_Carro`** (float): Valor de mercado / FIPE aproximado.
* **`Valor_Seguro`** (float): **Variável alvo (Target)** a ser prevista pela IA.

---

## 🧠 Desempenho da IA

* **Algoritmo**: `LinearRegression` (Scikit-Learn)
* **Acurácia na prova ($R^2$ Score)**: **~89,1%**
* **Tratamento de Categóricas**: One-Hot Encoding (`pd.get_dummies`) gerando mais de 200 colunas binárias.

---

## 🚀 Como Executar o Projeto

### 1. Clone o repositório ou acesse a pasta:
```bash
git clone https://github.com/SEU_USUARIO/NayarIA-5.git
cd NayarIA-5
```

### 2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### 3. Execute a IA:
```bash
python main.py
```
*(ou `python "NayarIA 5.py"`)*

A IA irá:
1. Carregar as 100.000 linhas do dataset.
2. Treinar o modelo de regressão linear.
3. Exibir no terminal os testes comparativos em cores.
4. Exibir a nota final obtida na prova de teste.
5. Abrir um prompt interativo para você digitar o seu salário, idade e carro para calcular o seguro na hora!

---

## 📝 Anotações do que Aprendi ao Longo da Jornada

### Fundamentos de Machine Learning:
* **Overfitting**: A IA decorou os dados de treino e não aprendeu a generalizar. É perceptível quando o modelo tem notas perfeitas no treino, mas erra feio nos dados de teste.
* **Underfitting**: A IA não conseguiu aprender o suficiente com os dados fornecidos e generalizou demais. Ex: acreditar que todo animal de 4 patas é um gato (desconsiderando cachorros, lobos, tigres). Resolve-se adicionando mais critérios (mais colunas relevantes) ou usando modelos mais adequados.
* **O Equilíbrio no ML**: Nem memorizar demais (overfitting), nem simplificar demais (underfitting). O objetivo é a boa capacidade de **generalização**.
* **Volume de Dados**: Uma IA precisa de volume de dados. Sem dados suficientes, ela não consegue captar padrões complexos e erra facilmente. Com 100.000 linhas, o modelo atingiu ~89% de acerto.
* **`random_state`**: Muito necessário para garantir **reprodutibilidade**. Garante que o sorteio de treino e teste seja o mesmo a cada execução.
* **`train_test_split`**: A ordem das variáveis retornadas é: `X_treino, X_teste, y_treino, y_teste` (onde `X` são as pistas/features e `y` é o gabarito/target).

### Tratamento e Engenharia de Dados:
* **`pd.get_dummies()`**: Indispensável para modelos que exigem dados puramente numéricos (como a Regressão Linear). Transforma variáveis textuais categóricas (como marcas e modelos) em colunas binárias de $0$ e $1$ (One-Hot Encoding).
* **Eixos no Pandas (`axis`)**:
  * `axis = 0`: Trata linhas.
  * `axis = 1` (ou `axis='columns'`): Trata colunas.
* **`.reindex(columns=..., fill_value=0)`**: O pulo do gato para colocar modelos em produção! Quando o usuário digita apenas um carro, o `get_dummies` só cria 1 coluna daquela marca. O `.reindex()`:
  1. Força a nova entrada a ter **todas as 200+ colunas** que a IA viu no treino.
  2. Preenche com **`0`** as outras marcas/modelos que não foram escolhidos (`fill_value=0`).
  3. Se o usuário digitar uma marca inédita, o código descarta a coluna desconhecida sem quebrar a aplicação.
* **`.strip()` e `.title()`**:
  * `.strip()`: Remove espaços em branco acidentais no início e fim do texto.
  * `.title()`: Padroniza o texto com as primeiras letras maiúsculas (ex: `"toyota"` $\rightarrow$ `"Toyota"`), evitando que a IA desconsidere a marca por diferença de maiúsculas/minúsculas.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3**
* **Pandas** (manipulação, ETL e One-Hot Encoding)
* **NumPy** (cálculos vetoriais e simulações estatísticas)
* **Scikit-Learn** (modelagem estatística, divisão treino/teste e Regressão Linear)
