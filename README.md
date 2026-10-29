# 🐦 Angry Birds treinado por Algoritmo Genético (Scilab)

Uma rede neural **MLP 5-8-2** aprende a jogar uma versão simplificada de Angry Birds **sem nenhum dado de treino rotulado e sem backpropagation**. Quem ajusta os pesos da rede é um **Algoritmo Genético (AG)**, e a única informação de aprendizado é o desempenho da rede no jogo (a *fitness*).

Projeto desenvolvido em **Scilab**, na disciplina de Inteligência Artificial do curso de Engenharia de Computação (Uniube).

---

## 📌 Ideia do projeto

Em vez de ensinar a rede com exemplos do tipo "para este cenário, o ângulo correto é X", o programa faz o seguinte:

1. A rede recebe a descrição de um cenário (onde está o porco e onde estão dois obstáculos).
2. A rede devolve **ângulo** e **velocidade** do lançamento.
3. Um simulador de física lança o pássaro e diz se ele acertou o porco, bateu num obstáculo ou caiu no chão.
4. Esse resultado vira a nota (fitness) da rede.
5. O AG mantém uma população de redes, seleciona as melhores, cruza e muta os pesos, gerando redes cada vez melhores.

Esse tipo de abordagem é conhecido como **neuroevolução**.

---

## 🧠 Como funciona

### Cromossomo
O cromossomo é o **vetor com todos os pesos e biases da rede**, ou seja, cada indivíduo da população é uma rede neural completa.

```
N_PESOS = 5·8 + 8 + 8·2 + 2 = 66 genes
```

### Rede neural (MLP 5-8-2)

| Camada | Neurônios | Ativação |
|---|---|---|
| Entrada | 5 | – |
| Oculta | 8 | `tanh` |
| Saída | 2 | sigmoide |

**Entradas** (normalizadas para ficarem na mesma escala):

| Entrada | Significado | Normalização |
|---|---|---|
| `px` | posição x do porco | ÷ 40 |
| `o1x` | posição x do obstáculo 1 | ÷ 40 |
| `o1h` | altura do obstáculo 1 | ÷ 5 |
| `o2x` | posição x do obstáculo 2 | ÷ 40 |
| `o2h` | altura do obstáculo 2 | ÷ 5 |

**Saídas**: a sigmoide gera valores em [0, 1], que são convertidos para as faixas físicas:

- Ângulo: de **10° a 80°**
- Velocidade: de **10 a 35 m/s**

### Simulação física

O lançamento segue o movimento de um projétil com gravidade, sem resistência do ar:

```
x(t) = v · cos(θ) · t
y(t) = y0 + v · sin(θ) · t − ½ · g · t²
```

| Parâmetro | Valor |
|---|---|
| Gravidade `g` | 9,8 m/s² |
| Altura inicial `y0` | 1,0 m |
| Raio do pássaro | 0,3 m |
| Raio do porco | 1,0 m (centro em y = 1,0) |
| Largura dos obstáculos | 1,0 m |
| Tempo simulado | 0 a ~8 s (400 passos de 0,02 s) |

O voo termina no **primeiro impacto**: chão ou obstáculo. Se o pássaro chegar à região do porco antes disso, conta como **acerto**.

### Geração de cenários

Cada cenário é sorteado aleatoriamente, de forma que seja sempre possível de resolver:

- Porco entre **20 e 40 m** de distância.
- Obstáculo 1 na primeira metade do caminho, obstáculo 2 na segunda metade (antes do porco).
- Alturas dos obstáculos entre **1,5 e 5 m**.

### Função de fitness

Para cada cenário:

| Situação | Fitness |
|---|---|
| Acertou o porco | `1.0` |
| Errou | `0.9 / (1 + d_min)`, onde `d_min` é a menor distância do voo ao porco |
| Errou **e** bateu em obstáculo | o valor acima × `0.5` (penalidade) |

A fitness de um indivíduo é a **média** dessa nota sobre todos os cenários da geração. A distância mínima dá um "gradiente" para o AG: lançamentos que passam perto do porco valem mais do que lançamentos que passam longe, mesmo sem acertar.

### Algoritmo Genético

| Componente | Configuração |
|---|---|
| População | 100 indivíduos |
| Gerações | 150 |
| Inicialização | pesos uniformes em [−2, 2] |
| Seleção | **torneio** (k = 3) |
| Cruzamento | **SBX** (Simulated Binary Crossover), η = 2, probabilidade 0,9 |
| Mutação | gaussiana, probabilidade 0,08 por gene, σ = 0,3 |
| Elitismo | os 4 melhores passam direto para a próxima geração |
| Cenários de treino | 30 por geração |

**Detalhe importante:** os cenários de treino são **sorteados de novo a cada geração**. Se fossem fixos, a rede poderia simplesmente decorar as respostas (overfitting) em vez de aprender a relação entre o cenário e o lançamento.

### Avaliação

Depois do treino, o melhor indivíduo é testado em **300 cenários inéditos** (semente diferente da usada no treino), e o programa reporta:

- % de cenários em que **acertou** o porco;
- % de cenários em que **bateu em obstáculo**.

---

## 📊 Saídas do programa

- **Log no console** a cada 10 gerações com a melhor fitness e a fitness média.
- **Resultado do teste** (taxa de acerto e de colisão).
- **Gráficos**:
  - evolução da fitness (melhor e média) ao longo das gerações;
  - 3 trajetórias de exemplo no conjunto de teste, mostrando o pássaro, os obstáculos (marrom), o porco (verde), o ângulo e a velocidade escolhidos pela rede e se acertou ou errou.
- **`melhor_pesos.sod`**: arquivo com os pesos da melhor rede, para reutilizar sem treinar de novo.

---

## ▶️ Como executar

**Requisitos:** [Scilab](https://www.scilab.org/download) (versão 6 ou superior recomendada).

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
   ```
2. Abra o Scilab e vá até a pasta do projeto.
3. Execute o script:
   ```scilab
   exec("angrybirdsAG.sce", -1)
   ```
   (ou abra o arquivo no SciNotes e clique em *Executar*).

O treino leva alguns instantes, pois cada geração avalia 100 redes em 30 cenários.

### Reutilizar a rede treinada

```scilab
load("melhor_pesos.sod");              // carrega a variável "melhor"
c = [30, 10, 3, 22, 2.5];              // [px, o1x, o1h, o2x, o2h]
[ang, vel] = mlp(melhor, entradas_rede(c));
```

### Ajustando os parâmetros

A chamada principal fica no final do arquivo:

```scilab
[melhor, hist] = treinar(n_pop, n_ger, n_elite, n_treino, semente);
//                         100    150     4        30        0
```

Experimente alterar o tamanho da população, o número de gerações, a taxa de mutação (`pm`, `sigma`) ou a arquitetura da rede (`N_HID`) e compare as curvas de fitness.

---

## 🗂️ Estrutura do código

| Função | O que faz |
|---|---|
| `gerar_cenario` | sorteia um cenário (porco + 2 obstáculos) |
| `entradas_rede` | normaliza o cenário para virar entrada da rede |
| `desempacotar` | transforma o vetor de pesos em matrizes `W1`, `b1`, `W2`, `b2` |
| `mlp` | forward da rede; devolve ângulo e velocidade |
| `simular` | simula o voo e detecta acerto, colisão e distância mínima |
| `fitness_cenario` / `fitness` | nota de um lançamento / média da rede nos cenários |
| `torneio` | seleção por torneio |
| `sbx` | cruzamento binário simulado |
| `mutacao` | mutação gaussiana |
| `treinar` | laço principal do AG |
| `avaliar` | teste em cenários inéditos |

---

## 📈 Resultados

> Preencha com os números obtidos na sua execução.

- Fitness do melhor indivíduo na última geração: `___`
- Taxa de acerto no teste (300 cenários): `___ %`
- Taxa de colisão com obstáculos no teste: `___ %`

*(Dica: adicione aqui um print do gráfico de evolução da fitness e das trajetórias.)*

---

## 💡 Possíveis melhorias

- Adicionar resistência do ar ou vento ao simulador.
- Incluir mais obstáculos ou porcos em alturas diferentes.
- Comparar com outros métodos (Estratégias Evolutivas, PSO, busca aleatória).
- Testar outras arquiteturas de rede e funções de ativação.
- Executar vários treinos com sementes diferentes e reportar média e desvio padrão.

---

## 🛠️ Tecnologias e conceitos

`Scilab` · `Algoritmos Genéticos` · `Redes Neurais (MLP)` · `Neuroevolução` · `SBX` · `Seleção por torneio` · `Elitismo` · `Simulação física`

---

## 👤 Autor

**Matheus Barcelos de Almeida**
Engenharia de Computação, Universidade de Uberaba (Uniube)

Projeto acadêmico da disciplina de Inteligência Artificial.
