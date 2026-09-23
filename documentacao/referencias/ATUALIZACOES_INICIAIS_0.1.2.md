# Atualizações e Diretrizes de Projeto — Versão 0.1.2
**Data de Revisão:** 19 de setembro de 2026  
**Status:** Documento de Especificação Técnica, Inovações Estratégicas e Métodos Numéricos  
**Referência Base:** Versão 0.1.1 (`kaggriculture_agent/`, 103 testes aprovados, snapshot do motor `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`)

---

## 1. Sumário Executivo da Evolução do Projeto

O projeto atingiu um salto substancial de maturidade técnica entre as versões 0.1.0 e 0.1.1:
- **0.1.0:** Agente restrito a 2 culturas (duplo ciclo), sem animais, sem compra de terra (`BUY_LAND`), confinado a 20 parcelas. Saldo médio contra o `starter`: **~21.063 moedas**.
- **0.1.1:** Implementação do Simplex primal de duas fases em Python padrão, inclusão das 5 culturas no portfólio, pecuária intensiva (até 8 vacas e 4 ovelhas), compra de terras, trava atômica de segurança para rega no mesmo dia e arena Bradley-Terry com 103 testes de regressão aprovados. Saldo médio contra o `starter`: **~113.824 moedas** (com picos de **137.773 moedas**), um salto de **+440% (5,4x)**.
- **0.1.2 (Esta Versão):** Transição de um modelo reativo avançado para um **agente ótimo predador e matemático**, incorporando:
  1. Soluções exatas para os gargalos remanescentes da 0.1.1 (Branch & Bound inteiro, regressão quantílica offline, valuation dual de fertilizante e contextual bandit).
  2. Estratégias competitivas não-convencionais (*Opponent Fingerprinting*, *Predatory Price Crashing*, *Wheat Squeeze*, *Town Shop Bayesian Forecasting* e *Bucket-Brigade*).
  3. Aplicação formal de **Cálculo Numérico** (Newton-Raphson marginal, Quadratura de Gauss-Hermite, Hessiana Diagonal separável e filtro de Savitzky-Golay).

---

## 2. Diagnóstico Residual e Oportunidades da Versão 0.1.1

Embora a v0.1.1 seja altamente competitiva, identificaram-se quatro fronteiras de melhoria que limitam o teto do agente:
1. **Simplex Contínuo vs. MILP Inteiro:** O Simplex contínuo resolve o poliedro em $\mathbb{R}^n$ e recorre a um arredondamento guloso viável (*greedy repair*). Isso não assegura ótimo inteiro global em restrições combinatórias justas de caixa e terra.
2. **Quantil Empírico vs. Regressão Quantílica Treinada:** O parâmetro de risco utiliza quantis empíricos de 32 cenários estocásticos internos com ruído gaussiano. Falta calibração histórica supervisionada a partir de replays reais da competição.
3. **Subaproveitamento Agronômico de Fertilizante:** 100% do fertilizante gerado pelos animais é vendido no mercado público. Em cenários de cotação deprimida de fertilizante, aplicá-lo em culturas de alto valor (Melão e Morango) gera receita líquida substancialmente superior à venda direta.
4. **Seleção Determinística de Especialistas:** A comutação entre perfis (`melon`, `cashflow`, `contrarian`, `liquidator`) ocorre por cortes fixos de dias e caixa, sem inferência bayesiana sobre o estilo de jogo do oponente.

---

## 3. Novas Implementações de Engenharia para a Versão 0.1.2

### 3.1. Branch & Bound (B&B) Nativo sobre o Simplex (MILP em Python Puro)
- **Mecanismo:** Acoplar uma rotina de ramificação e poda (*Branch and Bound*) ao solver Simplex contínuo existente em `kaggriculture_agent/optimization.py`.
- **Funcionamento:**
  1. Resolve o relaxamento linear contínuo na raiz.
  2. Se uma variável de lote $x_i$ for fracionária (ex.: $x_{\text{melão}} = 5,4$), criam-se dois nós filhos com restrições adicionais no tableau: $x_i \le 5$ e $x_i \ge 6$.
  3. Poda por inviabilidade, por limite superior (*upper bound pruning*) e por integralidade.
- **Complexidade:** Como o problema possui no máximo 8 variáveis inteiras (5 culturas + 3 animais) e o espaço de parcelas é limitado ($\le 40$), a árvore fecha a solução **estritamente inteira e matematicamente ótima em $< 3$ a $5$ ms**, respeitando com folga o limite de tempo do Kaggle.

### 3.2. Regressão Quantílica Treinada Offline via Perda Pinball
- **Mecanismo:** Substituição dos cenários heurísticos por um modelo linear regularizado treinado offline em replays públicos de alta pontuação.
- **Função de Perda:** Para um quantil pessimista $\tau \in \{0.15, 0.25\}$ (Value-at-Risk), minimiza-se a função *Pinball* com regularização L2:
  $$\min_{\boldsymbol{\beta}^{(\tau)}} \sum_{k} \rho_\tau\left( y_k - \mathbf{z}_k^T \boldsymbol{\beta}^{(\tau)} \right) + \lambda \|\boldsymbol{\beta}^{(\tau)}\|_2^2$$
  onde $\rho_\tau(u) = u(\tau - \mathbb{I}_{\{u < 0\}})$ e $\mathbf{z}_k$ é o vetor de estado (estoque corrente, lojas ativas, culturas adversárias observáveis, dia do episódio).
- **Execução Online:** O agente carrega apenas a matriz estática de coeficientes $\boldsymbol{\beta}^{(\tau)}$ via dicionário nativo. No turno de inferência, o cálculo da previsão de risco é um produto escalar em **$O(1)$ ($< 0,02$ ms)**, com zero variância estocástica.

### 3.3. Algoritmo de Valuation Dual de Fertilizante (Venda vs. Aplicação)
- **Mecanismo:** Decisão dinâmica de alocação de fertilizante no `scheduler.py` e `economy.py`.
- **Formulação Matemática do Retorno Marginal Líquido:**
  Para cada unidade de fertilizante disponível na fazenda, compara-se:
  - **Retorno da Venda Direta:** $R_{\text{venda}} = \text{price\_at}(\text{"FERTILIZER"}, I_{\text{fert}})$.
  - **Retorno da Aplicação em Campo:** Aplicar fertilizante dobra a taxa de ganho em culturas de ciclo único ou dobra as unidades por evento em culturas recorrentes durante 3 dias.
    $$R_{\text{campo}}(\text{cultura}) = \Delta \text{Unidades} \times P_{\text{cultura}} - \text{CustoAçãoTrabalho}$$
- **Critério de Decisão:**
  - Se $\max_{\text{plantas}} R_{\text{campo}} > R_{\text{venda}}$, agenda-se `FERTILIZE` na planta de maior retorno marginal (prioridade máxima para Melão entre dias 6–8 e Morango em fase de frutificação).
  - Caso contrário, a unidade de fertilizante é encaminhada ao galpão para ordem de `SELL`.

### 3.4. Contextual Bandit Bayesiano (Thompson Sampling) para Especialistas
- **Mecanismo:** Substituição das regras determinísticas de especialistas por um algoritmo de bandido contextual bayesiano leve.
- **Modelagem:**
  - Contexto $\mathbf{s}_t$: Vetor de pistas dos 3 primeiros dias (animais do oponente, sementes compradas, lojas abertas).
  - Cada braço especialista $k \in \{\text{adaptive, melon\_rusher, cashflow, contrarian, liquidator}\}$ possui uma distribuição normal a priori sobre a probabilidade de recompensa $\theta_k \sim \mathcal{N}(\mu_k, \sigma_k^2)$.
  - O meta-controlador amostra $\hat{\theta}_k$ e seleciona o especialista dominante, adaptando a conduta do agente sem necessidade de redes neurais pesadas.

---

## 4. Estratégias Não-Convencionais ("Fora da Caixa")

```
                      ESTRATÉGIAS NÃO-CONVENCIONAIS DA v0.1.2
                                         │
        ┌──────────────────┬─────────────┴─────────────┬──────────────────┐
        ▼                  ▼                           ▼                  ▼
  Opponent           Predatory Price             Wheat Starvation   Town Shop Option
Fingerprinting           Crashing                  & Squeeze            Pricing
(Identificação de    (Inundação de             (Guerra da Ração     (Antecipação de
Bots Estáticos)     Mercado no Joelho)           de Trigo)           Lojas Futuras)
```

### 4.1. *Opponent Fingerprinting* (Engenharia Reversa de Bots Estáticos)
- **Fundamento:** A auditoria do código de concorrentes de topo (como o V16-RC5) revelou que os melhores bots públicos utilizam trajetórias pré-gravadas estáticas de 720 turnos.
- **Detecção:** O agente monitora as ordens do adversário nos primeiros 5 turnos. Se o oponente executar a assinatura conhecida (ex.: compra de 1 vaca, 4 ovelhas, 14 trigos e 5 contratações no turno zero), o agente cataloga o adversário com precisão determinística.
- **Exploração:** Uma vez identificado o bot adversário, a trajetória futura de colheita e venda dele é 100% conhecida. O nosso agente sincroniza ordens de liquidação para ocorrerem **exatamente 1 turno antes do oponente**, colapsando o preço antes da chegada do lote concorrente.

### 4.2. *Predatory Price Crashing* (Inundação Predatória de Mercado)
- **Fundamento:** A curva oficial de produtos como **MELON** e **WOOL** possui comportamento quadrático (`"sq"`) quando o estoque ultrapassa $I_0 = 10.000$:
  $$P(I) = \text{base} - \text{amp} \times \left( \frac{I - I_0}{T} \right)^2$$
- **Execução:**
  1. O agente monitora visualmente a idade dos melões nas células públicas do adversário.
  2. Sabendo o dia exato em que o oponente colherá seu lote, o nosso agente acumula e liquida um lote intermediário imediatamente antes.
  3. O mercado entra na zona de saturação quadrática e a cotação recebida pelo adversário desaba de 250 para menos de 40 moedas por unidade, destruindo a rentabilidade de todo o ciclo de 10 dias do oponente.

### 4.3. *Wheat Starvation & Squeeze* (Bloqueio da Ração Animal)
- **Fundamento:** Qualquer animal (Vaca, Ovelha, Ganso) **foge e é eliminado da partida** se passar 2 dias consecutivos sem receber Trigo (`consecutive_unfed >= 2`).
- **Execução:**
  - Se o adversário mantiver um rebanho grande (ex.: 8C/4S) sem parcelas suficientes de trigo plantadas em sua fazenda, ele é 100% dependente da compra contínua de trigo via `BUY_PRODUCT WHEAT`.
  - O nosso agente monitora o estoque municipal de trigo. Caso o oponente enfrente pressão de caixa por compra de terras, o agente realiza compras táticas de trigo ou retém a oferta, empurrando o preço do trigo para cima.
  - A escassez e o encarecimento forçam o oponente a falhar na alimentação diária, provocando a fuga dos animais e liquidando o patrimônio adversário.

### 4.4. *Town Shop Bayesian Forecasting* (Precificação de Opções nas Lojas Futuras)
- **Fundamento:** As lojas municipais desbloqueiam a cada 3 dias (`townShopUnlockInterval = 3`), sorteadas com reposição a partir de uma lista fechada de 8 estabelecimentos (`SHOPS`).
- **Execução:**
  - Em vez de reagir às lojas apenas após o desbloqueio, o planejador calcula a esperança matemática do incremento de demanda para cada mercadoria:
    $$\mathbb{E}[\Delta D_i] = \sum_{s \in \text{SHOPS}} P(s \text{ abrir}) \times \text{consumo}(s, i)$$
  - Essa probabilidade condicional é incorporada ao valor presente das culturas no Simplex, permitindo antecipar plantios de culturas cuja demanda tem alta probabilidade de explosão nos ciclos seguintes.

### 4.5. Logística por "Esteira Humana" (*Bucket Brigade* Celular)
- **Fundamento:** Trabalhadores que caminham da periferia até o galpão gastam até 16 turnos apenas em trânsito de ida e volta.
- **Execução:**
  - Especialização funcional das unidades:
    - *Trabalhadores de Campo (Fixos):* Dedicados exclusivamente a regar, fertilizar e colher em quadrantes específicos, mantendo ocupação produtiva em 100% dos turnos.
    - *Mensageiros de Logística (Runners):* Unidades dedicadas a recolher excedentes e realizar depósitos no galpão central.
  - Como unidades podem compartilhar células sem colisão, o revezamento logístico economiza até 40% dos turnos ociosos de deslocamento.

---

## 5. Aplicações de Cálculo Numérico no Kaggriculture

O emprego formal de Análise e Cálculo Numérico substitui aproximações grosseiras por formulações exatas com convergência garantida:

```
                         TÉCNICAS DE CÁLCULO NUMÉRICO
                                       │
      ┌──────────────────┬─────────────┴─────────────┬──────────────────┐
      ▼                  ▼                           ▼                  ▼
Newton-Raphson na     Quadratura de             Otimização por         Filtro de
Receita Marginal     Gauss-Hermite             Hessiana Diagonal    Savitzky-Golay
(Lote Ótimo q*)     (Substitui Monte Carlo)      (O(n) Exato)       (Derivada dI/dt)
```

### 5.1. Zeros de Funções e Newton-Raphson: Otimização do Tamanho de Lote ($q^*$)
- **Problema:** Determinar a quantidade ótima $q^*$ a vender para maximizar o lucro líquido sem saturar o preço abaixo do custo de reserva $C$.
- **Formulação:**
  A receita acumulada de vender $q$ unidades é $R(q) = \int_0^q P(I + u) \, du$.
  A condição de primeiro grau para o lote ótimo requer que a Receita Marginal anule o ganho líquido:
  $$g(q) = P(I + q) - C = 0$$
- **Método de Newton-Raphson:**
  $$q_{k+1} = q_k - \frac{P(I + q_k) - C}{P'(I + q_k)}$$
  Como as derivadas analíticas $P'(I)$ das curvas oficiais ($\sqrt{\cdot}$, $\ln$, quadrática, linear, hinge) são estritamente contínuas e diferenciáveis por partes, o método atinge convergência com tolerância $10^{-6}$ em **2 a 3 iterações ($< 0,02$ ms)**.

### 5.2. Quadratura Gaussiana (Gauss-Hermite) Substituindo Simulação Monte Carlo
- **Problema:** A estimativa de preço e receita futuros via 32 cenários Monte Carlo possui erro estocástico de ordem $O(1/\sqrt{N})$, gerando ruído e oscilações artificiais entre decisões consecutivas.
- **Formulação:** Sob incerteza normal de oferta externa $S \sim \mathcal{N}(\mu, \sigma^2)$, o valor esperado da receita é dado pela integral:
  $$\mathbb{E}[R] = \int_{-\infty}^{+\infty} R(q \mid S) \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(S-\mu)^2}{2\sigma^2}} \, dS$$
- **Aproximação por Quadratura de Gauss-Hermite (5 Pontos):**
  $$\mathbb{E}[R] \approx \frac{1}{\sqrt{\pi}} \sum_{j=1}^{5} w_j \cdot R\left(q \;\middle|\; \mu + \sqrt{2}\sigma \xi_j\right)$$
  onde $\xi_j$ são as raízes de $H_5(x)$ e $w_j$ são pesos pré-calculados.
- **Ganho:** Precisão analítica equivalente a mais de 1.000 cenários de Monte Carlo, **completamente determinística**, executando **6 vezes mais rápido**.

### 5.3. Otimização Convexa Separável com Hessiana Diagonal em $O(n)$
- **Problema:** A curva de receitas de mercado é não linear e côncava. O Simplex exige linearização por fatias lineares (*tranches*).
- **Formulação:** O problema contínuo exato de alocação de culturas é:
  $$\max_{\mathbf{x}} \quad \Phi(\mathbf{x}) = \sum_{i=1}^{n} R_i(x_i) - \mathbf{c}^T \mathbf{x} \quad \text{sujeito a} \quad \mathbf{A}\mathbf{x} \le \mathbf{b}, \quad \mathbf{x} \ge 0$$
- **Propriedade Numérica Fundamental:**
  Como cada cultura afeta apenas o seu respectivo mercado, a função $\Phi(\mathbf{x})$ é aditivamente separável. Isso implica que a matriz Hessiana é **estritamente diagonal**:
  $$\nabla^2 \Phi(\mathbf{x}) = \operatorname{diag}\left( R''_1(x_1), R''_2(x_2), \dots, R''_n(x_n) \right)$$
- **Consequência:** A inversão da Hessiana no método de Newton com projeção KKT é realizada em tempo linear **$O(n)$** em vez de $O(n^3)$. A alocação contínua ótima converge em **$< 0,5$ ms**.

### 5.4. Diferenciação Numérica Suavizada via Filtro de Savitzky-Golay
- **Problema:** Calcular a derivada do estoque público $dI/dt$ por diferenças finitas simples amplifica o ruído gerado pelas compras pontuais das lojas da cidade.
- **Formulação:** Ajusta-se um polinômio local de grau 2 por mínimos quadrados sobre uma janela deslizante de 7 turnos.
- **Resultado:** A velocidade instantânea de esvaziamento/enchimento do mercado é obtida por convolução direta de coeficientes inteiros fixos:
  $$\left. \frac{dI}{dt} \right|_{t} = \frac{1}{28} \Big( -3 I_{t-3} - 2 I_{t-2} - 1 I_{t-1} + 0 I_t + 1 I_{t+1} + 2 I_{t+2} + 3 I_{t+3} \Big)$$
  Permite prever acelerações de venda do adversário e esgotamento de estoques com alta estabilidade numérica.

---

## 6. Matriz Comparativa de Arquitetura: 0.1.0 vs. 0.1.1 vs. 0.1.2

| Dimensão | Versão 0.1.0 (Legado) | Versão 0.1.1 (Atual) | Versão 0.1.2 (Alvo) |
| :--- | :--- | :--- | :--- |
| **Pontuação Típica** | ~21.000 moedas | ~113.800 a 137.000 moedas | **Meta: > 145.000 moedas** |
| **Otimização de Terras/Culturas** | Enumeração 2D heurística de 2 culturas | Simplex linear contínuo + reparo guloso | **Simplex com Branch & Bound (MILP exato)** |
| **Decisão de Lote de Venda** | Regras estáticas / Vender estoque total | Front-running de 1 turno empírico | **Newton-Raphson na Receita Marginal ($g(q)=0$)** |
| **Modelagem de Incerteza** | 32 cenários Monte Carlo não calibrados | Quantil empírico de cenários Monte Carlo | **Quadratura Gaussiana + Regressão Quantílica Offline** |
| **Uso de Fertilizante** | Inativo | 100% vendido no mercado | **Valuation Dual (Venda vs. Aplicação em Melão/Morango)** |
| **Comportamento Competitivo** | Cego ao oponente | Observa ocupação adversária para prior | **Opponent Fingerprinting + Predatory Price Crashing** |
| **Logística de Mão de Obra** | Atribuição gulosa plana | Zoneamento espacial e trava atômica de rega | **Zoneamento por Esteira Humana (Bucket-Brigade)** |
| **Conformidade Kaggle** | 100% Python stdlib | 100% Python stdlib | **100% Python stdlib (sem dependências externas)** |

---

## 7. Roteiro de Implementação Incremental para a Versão 0.1.2

1. **Etapa 1 (Cálculo Numérico e Inversão de Mercado):**
   - Implementar a rotina de Newton-Raphson em `kaggriculture_agent/market.py` para cálculo de lote ótimo de venda e substituir o loop de Monte Carlo pela Quadratura de Gauss-Hermite de 5 pontos.
2. **Etapa 2 (Otimização Inteira Exata):**
   - Adicionar o resolvedor Branch & Bound sobre o Simplex em `kaggriculture_agent/optimization.py`.
3. **Etapa 3 (Agronomia de Fertilizante):**
   - Incorporar a lógica de *Valuation Dual* em `kaggriculture_agent/scheduler.py` para autorizar a ação `FERTILIZE` em melões e morangos rentáveis.
4. **Etapa 4 (Inteligência Competitiva e Fingerprinting):**
   - Implementar o detector de assinaturas de adversários estáticos em `kaggriculture_agent/agent.py` para ativar contra-ataques de liquidação antecipada.
5. **Etapa 5 (Validação e Benchmark):**
   - Executar a arena multimodelo com os 103 testes de regressão existentes, medindo o ganho de rating Bradley-Terry antes da submissão final.
