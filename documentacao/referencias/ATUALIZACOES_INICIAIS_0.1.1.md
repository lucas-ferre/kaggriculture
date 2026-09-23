# Atualizações Iniciais — Versão 0.1.1
**Data de Revisão:** 19 de setembro de 2026  
**Status:** Documento de Análise Crítica e Diretrizes Arquiteturais (Sem alteração no código de produção)  
**Referência Base:** Versão 0.1.0 (`kaggriculture_agent/`, snapshot do motor `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`)

---

## 1. Sumário Executivo e Diagnóstico da Versão 0.1.0

A versão 0.1.0 estabeleceu uma base sólida, determinística e estritamente compatível com o ambiente Kaggle:
- **Submissão autossuficiente:** O runtime em `kaggriculture_agent/` depende exclusivamente da biblioteca padrão do Python (sem NumPy, SciPy ou solvers externos).
- **Contratos do motor verificados:** 50 testes de regressão aprovados, cobrindo o horizonte terminal no turno 718 (step 719 é terminal), validação atômica de sementes para evitar descarte total por overbooking, acesso exclusivo de `SELL` ao galpão privado (`private.shed`), e preservação de excedentes de inventário via `PLACE` em oposição ao descarte agressivo de `DROP`.
- **Limitações competitivas observadas:** O agente opera de forma excessivamente conservadora. Ao jogar contra o baseline `starter`, vence com folga (~21.000 moedas), mas termina com capital ocioso elevado, área territorial confinada ao quadrante inicial de 5×5 e sem explorar pecuária ou fertilização.

---

## 2. Análise Crítica: Pontos de Melhoria e Riscos Lógicos

### 2.1. Gargalos Estratégicos de Alto Impacto
1. **Paralisia Territorial e Subaproveitamento de Capital:**
   - O agente nunca emite a ordem `BUY_LAND`. O primeiro quadrante extra custa 1.000 moedas, enquanto o agente acumula entre 20.000 e 40.000 moedas paradas na carteira.
   - Consequência: Fica limitado a ~20 lotes produtivos no quadrante NW inicial, gerando um gargalo de escala severo frente a oponentes expansivos.
2. **Ausência de Animais e Economia de Fertilizante:**
   - No snapshot oficial, construir estruturas (`BUILD_COOP`, `BUILD_PASTURE`) consome apenas uma ação física da unidade, sem custo monetário de construção.
   - Gansos, vacas e ovelhas geram produtos de alto valor agregado com demanda constante no mercado e fornecem fertilizante diário sem custo adicional. O fertilizante dobra o ganho diário das culturas ou pode ser comercializado por preço-base de 100 moedas. Esse subsistema está completamente inativo no agente.
3. **Rigidez Estrutural do Planejador ("Duplo Ciclo" Arbitrário):**
   - O `CropPlanner` força a alocação de exatamente uma cultura rápida (Trigo ou Cenoura) e uma cultura de ciclo longo (Tomate ou Morango), mantendo uma meta fixa de área rápida (`fast_share = 0.6`).
   - Essa heurística impede a formação de portfólios superiores, como monocultura de Melão quando as cotações estão favoráveis (que elevou o saldo de 21k para 40.6k nos experimentos locais), diversificação em 3 culturas ou rotação focada na demanda das lojas recém-desbloqueadas na cidade.

### 2.2. Riscos Lógicos e Fragilidades de Escalonamento
1. **Viés de Agregação de Lote Único (*Batch Slippage Bias*):**
   - Ao projetar a receita futura no `CropPlanner`, todas as colheitas ao longo do horizonte são consolidadas em um único lote agregado vendido em uma data média ponderada.
   - Como a curva de preço oficial penaliza grandes volumes vendidos de uma vez só (*market slippage*), o planejador superestima drasticamente a perda de preço de culturas recorrentes, pois na execução prática as colheitas são fracionadas e vendidas diariamente em lotes pequenos.
2. **Risco de Perda Noturna de Sementes Recém-Plantadas:**
   - Plantas novas iniciam com `consecutive_unwatered = 1`. Caso não recebam `WATER` no mesmo dia de calendário, na virada da noite (hora 23 para hora 0) o contador atinge 2 e a semente morre, transformando-se em `WEED`.
   - O agendador permite a emissão de `PLANT` até `day_left - 1`. Se o plantio for concluído na hora 22, a tarefa de rega só é criada na hora 23; caso as unidades estejam distantes ou ocupadas com colheitas prioritárias, a semente é perdida. É indispensável implementar agendamento atômico em par (`PLANT` + `WATER` garantido).
3. **Distorção no Modelo Estocástico de Mercado:**
   - O `MarketModel` modela a concorrência adversária como uma taxa de fluxo contínua via decaimento exponencial (EMA) com ruído Gaussiano em 32 cenários Monte Carlo.
   - Na prática, adversários humanos e bots avançados descarregam mercadorias em pulsos discretos (degraus de colheita). A média móvel contínua atenua picos de oferta e falha em antecipar quebras repentinas de mercado decorrentes da maturação de safras visíveis no campo adversário.
4. **Ineficiência Espacial e Oscilação de Rotas:**
   - A atribuição estática do turno (via algoritmo Húngaro ou guloso) minimiza apenas a distância imediata `dist + 1`. Sem zoneamento territorial, trabalhadores frequentemente cruzam a fazenda em sentidos opostos, gerando perda de turnos úteis por deslocamento desnecessário.

---

## 3. Avaliação Técnica: Proposta 1 — Método Simplex + Regressão Quantílica

### 3.1. Conceito e Formulação Matemática
A proposta substitui a enumeração discreta heurística por um modelo de **Programação Linear Inteira Mista (MILP)** com garantia de otimalidade matemática para a alocação da fazenda:

1. **Variáveis de Decisão:**
   - $x_i \ge 0$: Número de parcelas de terra alocadas para a atividade/cultura $i \in \{\text{WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, GOOSE, COW, SHEEP}\}$.
2. **Função Objetivo com Regressão Quantílica:**
   - Em vez de maximizar a média de lucro (que expõe o agente ao risco de cauda quando o oponente inunda o mercado), o preço de venda projetado é obtido via **Regressão Quantílica** no quantil pessimista $\tau$ (ex.: $\tau = 0.15$ ou $\tau = 0.25$, correspondendo ao Value-at-Risk):
     $$\hat{P}_i(\tau \mid \mathbf{z}) = \beta_0^{(\tau)} + \mathbf{z}^T \boldsymbol{\beta}^{(\tau)}$$
     Onde $\mathbf{z}$ é o vetor de estado do mercado (estoque público, lojas ativas, culturas visíveis do adversário, dia atual).
   - O coeficiente de retorno líquido unitário de cada cultura passa a ser:
     $$c_i(\tau) = \hat{P}_i(\tau \mid \mathbf{z}) \cdot Y_i - \text{CustoSemente}_i - \text{CustoAçõesTrabalho}_i$$
   - Função Objetivo:
     $$\max_{\mathbf{x}} \sum_{i} c_i(\tau) \cdot x_i$$
3. **Restrições Lineares do Sistema Agrícola:**
   - **Terra Física:** $\sum_i x_i \le \text{ParcelasDesbloqueadas}$
   - **Orçamento de Mão de Obra (Ações/Dia):** $\sum_i a_i \cdot x_i \le N_{\text{trabalhadores}} \times 24 \times \text{eficiência}$
   - **Restrição de Liquidez (Caixa Imediato):** $\sum_i s_i \cdot x_i \le \text{DinheiroDisponível} - \text{Reserva}$
   - **Capacidade do Galpão:** $\sum_i d_i \cdot x_i \le \text{ShedCapacity}$

### 3.2. Tratamento da Curva de Preço Não Linear (*Piecewise Linear Tranches*)
Como o motor oficial reduz a cotação conforme o volume vendido aumenta, o preço não é puramente linear em $x_i$. A solução clássica em pesquisa operacional é dividir a produção de cada cultura em tranches por partes:
- Tranche 1 (1 a 5 parcelas): Preço de escassez $c_{i,1}$
- Tranche 2 (6 a 12 parcelas): Preço de equilíbrio $c_{i,2}$
- Tranche 3 (13+ parcelas): Preço com desconto por saturação $c_{i,3}$

Como a receita acumulada é côncava, o Simplex preenche naturalmente as variáveis da tranche de maior margem antes de avançar para a seguinte, mantendo a formulação 100% linear.

### 3.3. Viabilidade e Integração no Projeto
- **Conformidade Kaggle:** Totalmente viável. Um algoritmo Simplex Primal Duas-Fases em Python padrão requer cerca de 100 a 120 linhas de código limpo, sem pacotes externos.
- **Performance:** Para 8 a 15 variáveis e 6 restrições, o Simplex converge em menos de **2 milissegundos**, sendo significativamente mais rápido e completo do que varreduras combinatórias.
- **Execução da Regressão Quantílica:**
  - O ajuste dos coeficientes $\boldsymbol{\beta}^{(\tau)}$ deve ser realizado **offline** (treinado em base de replays e partidas de simulação).
  - Em tempo de execução dentro da partida, o cálculo do quantil se resume a multiplicações escalares $O(1)$ dos coeficientes pré-calculados pelo vetor de observação corrente.

---

## 4. Avaliação Técnica: Proposta 2 — Modelo Baseado em Células e Ensembles Dinâmicos

### 4.1. Abordagem Celular e Espacial (*Cellular & Grid Dynamics*)
A abordagem celular moderniza a camada tática de execução (`TaskScheduler`):

1. **Zoneamento Celular Funcional:**
   - A fazenda é dividida espacialmente por tipo de atividade:
     - *Núcleo Logístico (Raio 1 do Galpão):* Culturas que requerem colheita contínua (Tomate, Morango) e instalações de animais (reduzindo turnos gastos carregando produtos).
     - *Periferia de Colheita Única:* Culturas de ciclo fechado (Melão, Trigo), onde os trabalhadores realizam deslocamento pontual apenas no plantio e no dia de colheita.
     - *Corredores de Circulação:* Mantidos para evitar dispersão desordenada.
2. **Navegação por Campos de Potencial Celular (*Potential Fields*):**
   - Cada célula $(x, y)$ da grade emite um vetor de atração proporcional à sua urgência:
     - Planta necessitando de água: atração inversamente proporcional ao tempo até a meia-noite.
     - Produto pronto para colheita ou galpão com espaço para descarregar: gradiente gravitacional em direção ao celeiro.
   - As unidades movem-se seguindo a descida do gradiente local, eliminando o custo computacional do Húngaro e prevenindo o problema de rotas que se cruzam ou se cancelam entre turnos consecutivos.
3. **Simulação Celular Preditiva (Micro-Forward Simulation):**
   - Um modelo de transição estado-a-estado que projeta os próximos 24 turnos da própria fazenda. Isso garante matematicamente que nenhuma semente seja plantada sem que haja um slot garantido de rega antes da virada do dia.

### 4.2. Ensembles Dinâmicos (*Dynamic Ensembles*)

#### A. Na Jogatina (Runtime Meta-Controller)
Em vez de depender de um conjunto estático de parâmetros (`PolicyConfig`), o agente opera com um pool de subpolíticas especializadas:
- **Especialista 1 (Melon / High Margin):** Focado em acumulação rápida de terra e melão com baixa demanda de rega/trabalho.
- **Especialista 2 (Cashflow Intensivo):** Horta rápida (Trigo/Cenoura) associada à pecuária (Gansos/Ovos) para geração contínua de fluxo de caixa.
- **Especialista 3 (Contrarian / Arbitragem):** Monitora o quadrante do adversário; se o oponente investir massivamente em Trigo/Tomate, o especialista migra 100% da produção para culturas não concorrentes.
- **Especialista 4 (Endgame Liquidator):** Assume o controle nos últimos 4 dias para desativar novos plantios, colher tudo o que estiver maduro e liquidar estoques ao longo das melhores janelas de preço.

O **Meta-Controlador Dinâmico** (estruturado como um *Contextual Bandit* leve) avalia nos primeiros 3 dias quais lojas abriram na cidade e a postura do oponente, selecionando dinamicamente a subpolítica ou combinando suas ações por votação ponderada.

#### B. Na Bancada de Testes (Arena de Avaliação)
- Atualmente, a suíte de benchmarks utiliza apenas o `starter` e autojogo (`selfplay`).
- A bancada de ensembles cria uma **liga interna de adversários sintéticos** com diferentes personalidades estratégicas (agressivo, passivo, inundador de mercado, focado em animais).
- O desempenho passa a ser ranqueado pelo algoritmo **Bradley-Terry** (o exato modelo probabilístico usado pelo Kaggle na avaliação oficial pós-prazo), permitindo medir a robustez do agente contra estilos de jogo variados antes do envio final.

---

## 5. Estudo de Caso e Benchmark de Referência: Agente V16-RC5 (8C/4S)

### 5.1. Contexto e Resultados no Motor Oficial
Durante o ciclo de análise, auditamos o notebook público de alta pontuação [V16-RC5 | High-Score 8C/4S Premium Market Lead](https://www.kaggle.com/code/boatlee/v16-rc5-high-score-8c-4s-premium-market-lead/notebook), de autoria de **boatlee**. O agente foi avaliado em partidas completas de 720 turnos contra o motor oficial fixado:

| Confronto Direto (720 turnos) | Vencedor | Saldo do Vencedor | Saldo do Perdedor | Margem |
| :--- | :--- | :--- | :--- | :--- |
| **Boatlee (V16-RC5)** vs **Starter** | Boatlee | **121.478 moedas** | 3.692 moedas | +117.786 |
| **Boatlee (V16-RC5)** vs **Nosso Agente (v0.1.0)** | Boatlee | **137.773 moedas** | 24.465 moedas | +113.308 |

O resultado evidenciou experimentalmente a barreira de pontuação da nossa versão inicial: enquanto nosso agente atingiu ~24,5k moedas, o competidor de referência ultrapassou **137k moedas** (~5,6 vezes mais receita).

### 5.2. Desconstrução Estrutural da Estratégia V16-RC5
A descompilação do fluxo de 720 turnos revelou os quatro mecanismos responsáveis pela disparidade de pontuação:

1. **A Máquina Pecuária "8C/4S" (8 Vacas / 4 Ovelhas):**
   - No **turno zero**, o agente já adquire 1 Vaca, 4 Ovelhas, 5 sementes de Melão, 5 de Trigo, 14 trigos para alimentação e contrata 5 trabalhadores.
   - Até o **step 192** (dia 8), estabelece a meta completa de 8 Vacas e 4 Ovelhas.
   - Isso garante uma geração ininterrupta de **Leite** (base 160) e **Lã** (base 200), alimentada por Trigo e com demanda estável nas lojas da cidade.
2. **Expansão Territorial Agressiva (`BUY_LAND`):**
   - O agente emite duas ordens de compra de terra nos estágios iniciais, saindo do limite de 20 parcelas para operar em múltiplos quadrantes.
3. **Monetização Contínua de Fertilizante:**
   - Foram identificadas **55 ordens de venda de fertilizante** durante a partida. Como cada animal sobrevivente gera 1 fertilizante gratuito por dia, o produto é sistematicamente liquidado a ~100 moedas/unidade, constituindo uma renda passiva expressiva.
4. **Camada de Reparo de Ervas Daninhas (`_weed_repair_action`):**
   - O agente baseia-se em um plano de ações pré-computado (`_ACTIONS`). Para mitigar a estocasticidade do surgimento de ervas daninhas (`weedSpawnChance = 0.005`), ele intercepta intenções de construir pasto ou plantar em células infestadas, emite `DIG`, posterga o plano da unidade em 1 turno e aplica um buffer de 8 passos para ressincronização.
5. **Antecipação Tática de Mercado (*One-turn Premium Market Lead*):**
   - Se o cronograma planeja vender um lote no turno $t+1$, mas no turno $t$ o galpão já possui estoque e a cidade não consumiu mercadoria (`town_demand = 0`), ele **antecipa a venda no turno $t$** (`_front_run`).
   - No turno $t+1$, a função `_repay` abate essa quantidade da ordem regular.
   - Efeito: Ele "fura a fila" do mercado, capturando o preço pré-venda mais alto antes que ações adversárias ou quedas de cotação ocorram.

### 5.3. Vulnerabilidades do Boatlee e Vantagem Competitiva da nossa Arquitetura
Apesar da pontuação elevada, o V16-RC5 possui uma fragilidade arquitetural severa:
- **Operação Quase Open-Loop:** Ele executa uma sequência pré-gravada de 720 turnos. Se um adversário competitivo notar a composição 8C/4S nos primeiros dias e inundar o mercado com Leite, Lã ou Melão, o V16-RC5 **não consegue alterar sua produção**. Ele continuará investindo e colhendo os mesmos produtos com margens achatadas.
- **Oportunidade para a nossa Versão 0.1.1:** Ao incorporar a pecuária (8C/4S), o melão, a compra de terras e o *front-running* ao nosso planejador matemático (**Simplex + Regressão Quantílica** e **Ensembles Dinâmicos**), obteremos o mesmo poder bruto de geração de receita (>130k), mas com **capacidade reativa em tempo real**, superando bots estáticos no torneio final.

---

## 6. Matriz Comparativa de Arquitetura

| Dimensão | Versão 0.1.0 (Atual) | Agente Referência (Boatlee V16-RC5) | Versão 0.1.1 Alvo (Simplex + Ensembles) |
| :--- | :--- | :--- | :--- |
| **Pontuação Típica** | ~21.000 a 25.000 | **~120.000 a 138.000** | **Meta: > 140.000** |
| **Pecuária e Terra** | Bloqueadas (0 animais, 0 expansão) | 8 Vacas, 4 Ovelhas, 2 expansões de terra | Alocação livre de animais e terra via solver |
| **Planejamento** | Enumeração 2D heurística de 2 culturas | Trajetória pré-gravada (estática) | **Otimização Global Linear (Simplex nativo)** |
| **Execução Tática** | Distância Manhattan gulosa/Húngaro | Trajetória pré-calculada + Weed Repair | **Zoneamento Celular + Trava de Rega Segura** |
| **Interação de Mercado** | Venda no dia seguinte se previsão subir | **Front-Running (venda 1 turno adiantada)** | Previsão por Quantil + Front-Running adiantado |
| **Adaptabilidade** | Reavalia a cada 6 turnos (baixa amplitude) | Nula (produção física rígida) | **Alta (Ensemble Contrarian reage ao oponente)** |
| **Conformidade Kaggle** | 100% Python stdlib | 100% Python stdlib (base85 zlib) | 100% Python stdlib (zero dependências) |

---

## 7. Próximos Passos Recomendados para a Versão 0.1.1

1. **Camada de Pecuária e Terras:** Habilitar ordens `BUY_LAND`, `BUILD_PASTURE`, `BUY_ANIMAL` (Vacas e Ovelhas) e coleta/venda de fertilizante no `agent.py`.
2. **Mecanismo de Venda Antecipada (*Front-Running*):** Incorporar a lógica de *market lead* de 1 turno no `MarketModel` para produtos premium (Melão, Leite, Lã, Morango).
3. **Solver Simplex Nativo:** Substituir a busca combinatória de 2 culturas em `planner.py` por um solver Simplex Two-Phase em `optimization.py` para alocar terra, animais e mão de obra de forma ótima.
4. **Agendamento Celular com Trava de Rega:** Eliminar o risco de morte de sementes garantindo que nenhum `PLANT` seja emitido sem um `WATER` agendado e reservado no mesmo dia.
5. **Arena de Testes Multimodelo:** Expandir `scripts/benchmark.py` com múltiplos perfis de adversários avaliados pelo modelo probabilístico Bradley-Terry.
