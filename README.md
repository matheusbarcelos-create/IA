# IA

🐦 Angry Birds simplificado treinado por **Algoritmo Genético + MLP (5-8-2)** em **Scilab**.

Uma rede neural MLP aprende a jogar sem dados rotulados e sem backpropagation. Os pesos são ajustados por um AG, usando apenas o desempenho no jogo (fitness).

Projeto da disciplina de Inteligência Artificial (Engenharia de Computação, Uniube).

## 📌 Ideia do projeto

Em vez de ensinar com exemplos rotulados, o sistema:

1. recebe um cenário com porco e dois obstáculos;
2. prevê ângulo e velocidade de lançamento;
3. simula o voo do projétil;
4. atribui fitness ao lançamento;
5. evolui os pesos com AG (seleção, cruzamento, mutação e elitismo).

Essa abordagem é **neuroevolução**.

---

## 🧠 Como funciona

### Cromossomo

Cada indivíduo do AG representa todos os pesos e biases da rede.

\[
N_{PESOS} = 5\cdot8 + 8 + 8\cdot2 + 2 = 66
\]

### Rede neural (MLP 5-8-2)

| Camada | Neurônios | Ativação |
|---|---:|---|
| Entrada | 5 | – |
| Oculta | 8 | tanh |
| Saída | 2 | sigmoide |

Entradas normalizadas:

| Entrada | Significado | Normalização |
|---|---|---|
| px | posição x do porco | ÷ 40 |
| o1x | posição x do obstáculo 1 | ÷ 40 |
| o1h | altura do obstáculo 1 | ÷ 5 |
| o2x | posição x do obstáculo 2 | ÷ 40 |
| o2h | altura do obstáculo 2 | ÷ 5 |

Saídas convertidas para faixas físicas:

- **Ângulo:** 10° a 80°
- **Velocidade:** 10 a 35 m/s

### Simulação física

Sem resistência do ar:

\[
x(t)=v\cos(\theta)t
\]
\[
y(t)=y_0+v\sin(\theta)t-\frac{1}{2}gt^2
\]

Parâmetros:

- Gravidade: **9,8 m/s²**
- Altura inicial: **1,0 m**
- Raio do pássaro: **0,3 m**
- Raio do porco: **1,0 m**
- Obstáculo: largura **1,0 m**
- Simulação: ~8 s (400 passos de 0,02 s)

O voo termina no primeiro impacto (chão ou obstáculo). Se atingir a região do porco antes disso, conta como acerto.

### Geração de cenários

- Porco entre **20 e 40 m**
- Obstáculo 1 na primeira metade do caminho
- Obstáculo 2 na segunda metade (antes do porco)
- Alturas entre **1,5 e 5 m**

### Fitness

Para cada cenário:

- Acertou porco: **1.0**
- Errou: **0.9 / (1 + d_min)**
- Errou + colidiu em obstáculo: penalização × **0.5**

A fitness final é a média em todos os cenários da geração.

### Algoritmo Genético

| Componente | Configuração |
|---|---|
| População | 100 |
| Gerações | 150 |
| Inicialização | pesos uniformes em [-2, 2] |
| Seleção | torneio (k=3) |
| Cruzamento | SBX (η=2, p=0,9) |
| Mutação | gaussiana (p=0,08 por gene, σ=0,3) |
| Elitismo | 4 melhores |
| Cenários/geração | 30 |

Os cenários de treino são regenerados a cada geração para reduzir overfitting.

---

## 📊 Saídas do programa

- Log a cada 10 gerações (melhor e média)
- Resultado do teste em 300 cenários inéditos:
  - taxa de acerto
  - taxa de colisão em obstáculos
- Gráficos:
  - evolução da fitness
  - 3 trajetórias de exemplo com cenário, parâmetros e resultado
- Arquivo `melhor_pesos.sod` com os pesos finais

---

## ▶️ Como executar

Requisitos: **Scilab 6+**.

1. Clone:

```bash
git clone https://github.com/matheusbarcelos-create/IA.git
cd IA
```

2. No Scilab, execute:

```scilab
exec("angrybirdsAG.sce", -1)
```

O treino avalia 100 redes em 30 cenários por geração.

### Reutilizar rede treinada

```scilab
load("melhor_pesos.sod");
c = [30, 10, 3, 22, 2.5];
[ang, vel] = mlp(melhor, entradas_rede(c));
```

### Ajustar parâmetros

No final de `angrybirdsAG.sce`:

```scilab
[melhor, hist] = treinar(n_pop, n_ger, n_elite, n_treino, semente);
//                         100    150     4        30        0
```

Experimente alterar população, gerações, mutação ou arquitetura.

---

## 🗂️ Estrutura do código

- `gerar_cenario`: gera cenário válido
- `entradas_rede`: normaliza entradas
- `desempacotar`: monta pesos/biases da MLP
- `mlp`: forward e conversão para ângulo/velocidade
- `simular`: dinâmica do projétil e colisões
- `fitness_cenario` / `fitness_individuo`: avaliação
- `torneio`: seleção
- `sbx`: cruzamento
- `mutacao`: mutação gaussiana
- `treinar`: loop evolutivo
- `avaliar`: teste em cenários inéditos

---

## 📈 Resultados

Preencha com os números da sua execução:

- Fitness do melhor indivíduo na última geração: ___
- Taxa de acerto no teste (300 cenários): ___ %
- Taxa de colisão com obstáculos no teste: ___ %

---

## 💡 Possíveis melhorias

- Resistência do ar / vento
- Mais obstáculos e variações de altura
- Comparar com ES, PSO, busca aleatória
- Testar outras arquiteturas e ativações
- Rodar múltiplas sementes (média e desvio padrão)

---

## 🛠️ Tecnologias e conceitos

Scilab · Algoritmos Genéticos · MLP · Neuroevolução · SBX · Torneio · Elitismo · Simulação física

## 👤 Autor

Matheus Barcelos de Almeida  
Engenharia de Computação, Universidade de Uberaba (Uniube)
