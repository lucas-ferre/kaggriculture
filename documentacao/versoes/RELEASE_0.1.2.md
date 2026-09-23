# Atualização 0.1.2 — implementação e limites das propostas

A versão 0.1.2 incorpora os mecanismos verificáveis dos documentos [Atualizações e Diretrizes 0.1.2](../referencias/ATUALIZACOES_INICIAIS_0.1.2.md) e Análise do benchmark Top 1 (referência histórica). A implementação foi ajustada aos contratos do motor fixado, à informação pública disponível e ao orçamento computacional. Metas de pontuação, tempos de execução e superioridade teórica mencionados nas propostas não são tratados como resultados demonstrados.

O runtime continua integralmente Python stdlib. O motor de desenvolvimento permanece no commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`, sem alterações. Os pacotes originais da 0.1.0 e da 0.1.1 estão preservados para comparações reativas; a 0.1.2 não incorpora fitas nem código dos notebooks citados nos documentos.

## Configuração entregue

A política padrão mantém `strategy="adaptive"`, oito trabalhadores no total como teto, 40 parcelas, reserva de 300 moedas, até oito vacas, quatro ovelhas e duas expansões. Os limites não obrigam compras ou contratações sem retorno.

As novidades padrão são `optimizer="branch_bound"`, `integer_node_limit=64`, `market_method="quadrature"`, `enable_fertilizer=True` e `enable_opponent_model=True`. A seleção contextual de especialistas e a reserva tática adicional de trigo permanecem experimentais e desativadas: `enable_contextual_bandit=False` e `tactical_wheat_reserve=False`.

## Reconciliação das propostas de otimização e cálculo numérico

| Proposta | Implementação e estado | Alcance real |
| --- | --- | --- |
| Branch & Bound nativo | Ativo por padrão sobre Simplex de duas fases. Mantém solução inteira viável, limite superior, gap e orçamento de nós. | `optimal` exige fechamento da árvore dentro da tolerância numérica. Ao atingir limite, conserva incumbente e informa gap aberto. Não há garantia de fechamento em 3–5 ms. |
| MILP de todos os recursos | Tranches inteiras no portfólio agrícola e seleção inteira no investimento animal. | A pecuária reserva recursos antes do cultivo; mão de obra é comparada por equipes. Não há um único problema conjunto de animais, culturas, terra, mercado e rotas. |
| Custo marginal Fibonacci | O plano compara capacidade de equipes, contratação imediata e custo das contratações futuras modeladas. Reserva caixa da contratação antes das sementes. | O custo segue a sequência do motor; produtividade e deslocamento ainda são aproximados. Cuidar do patrimônio existente é compromisso, e déficits inevitáveis são registrados. |
| Regressão quantílica pinball/L2 | Pipeline offline implementado, com aprovação por produto e inferência por coeficientes locais. **Somente `WOOL` e `FERTILIZER` passaram o filtro inicial e estão no modelo embarcado.** | Amostra local pequena, quatro replays de treino e quatro de validação. A aprovação não demonstra ganho competitivo nem valida todas as distribuições futuras. |
| Gauss–Hermite de cinco pontos | Padrão ativo, com nós e pesos normalizados para uma aproximação normal do fluxo externo. Médias, probabilidades e quantis usam pesos verdadeiros. | Integra uma aproximação de um fator; não é solução exata do jogo, nem equivalente por definição a milhares de cenários. A alternativa `scenarios` mantém 32 cenários para ablação. |
| Newton–Raphson para lote ótimo | Substituído por bisseção inteira em `sale_quantity_at_reserve`, usando a cotação oficial. | O preço arredondado e com piso não oferece a derivada contínua presumida na proposta. A bisseção encontra o maior lote cuja unidade marginal alcança o preço de reserva, para uma venda isolada. |
| Hessiana diagonal e ótimo em O(n) | Não incorporado como solver alternativo. Mantiveram-se tranches lineares e B&B limitado. | Separabilidade do objetivo não elimina o acoplamento das restrições de recursos. Inverter uma Hessiana diagonal não resolve, por si só, o sistema KKT nem assegura o tempo/ótimo proposto. |
| Savitzky–Golay de sete amostras | Tendência quadrática causal na extremidade da janela, combinada ao histórico de resíduos corrigidos. | Usa apenas amostras já observadas. A fórmula centrada com observações futuras não é utilizável online; lacunas e censura no piso interrompem o uso do filtro. |
| Antecipação de lojas | Demanda futura esperada calculada sob a distribuição pública e calendário de desbloqueios; cenários alternativos podem amostrar lojas. | Não é identificação da futura loja nem aprendizado bayesiano de um parâmetro oculto. |

No plano agrícola, os **64 nós são compartilhados entre as opções de equipe**, com caches de previsão e valor reutilizados. A seleção animal faz sua própria chamada limitada. O número não deve ser interpretado como limite total de LPs de todo o turno.

O certificado do solver se refere ao problema linear inteiro efetivamente resolvido. O valor de receitas por tranches, a comparação de equipes, a execução e as previsões continuam aproximações. O preenchimento ou reparo posterior é viável, mas não transforma o conjunto do agente em um jogo globalmente ótimo.

## Quantis treinados: uso seletivo e rastreável

O experimento inicial está registrado em [quantile_calibration_0.1.2.json](../../outputs/quantile_calibration_0.1.2.json). O treino utiliza quatro replays da 0.1.1 congelada contra a 0.1.0, com seeds 101 e 103 e os dois lados. A validação utiliza quatro replays contra `starter`, com seeds 211 e 223 e os dois lados. Os atributos são públicos no instante de decisão; somente o alvo consulta o preço futuro.

O ajuste usa padronização obtida no treino, perda pinball, L2 e otimização coordenada limitada. A aprovação exige quantidade mínima de exemplos, melhora de pinball frente ao modelo empírico e cobertura dentro da tolerância registrada. Essa validação é um filtro de implantação, não um teste final intocado; lados pareados e horizontes sobrepostos introduzem correlação.

O arquivo embarcado contém apenas **lã e fertilizante**, para quantil 0,25 e horizonte de 24 turnos. Inferência exige esquema, produto, horizonte, quantil e faixa de atributos compatíveis. Fora dessas condições, a distribuição empírica continua ativa. A origem da previsão é observável em `last_quantile_source`.

O objeto aprendido é um **preço futuro isolado**. Quantis da receita de vários lotes cronológicos continuam calculados sobre trajetórias ponderadas; não se substitui o quantil da soma por uma soma de quantis de preço. O modelo também não elimina a incerteza sobre oponentes e execução.

## Reconciliação das estratégias competitivas e dos reflexos do benchmark

| Proposta ou mecanismo citado | Implementação e estado | Limite ou hipótese corrigida |
| --- | --- | --- |
| Valuation dual de fertilizante | Ativa: compara produção da planta com/sem aplicação, tetos, calendário, prazo e custo das ações com a alternativa de venda. | A cotação atual é uma hipótese de valorização. Melão ou recorrente no teto pode não ganhar unidades; a aplicação não é automática por cultura ou idade. |
| `SL2`: menos mãos sem retorno | Equipes comparadas por capacidade e Fibonacci; contratação respeita o alvo escolhido, liquidez e trabalho final disponível. | Não reproduz uma fita de contratação do concorrente nem garante ocupação integral das unidades. |
| `VE1` / `VT1`: ovelhas e encerramento | Calendários por data real de colocação, horizonte nas aquisições e supressão de ações sem pagamento futuro. | O dia 11 pode produzir um ciclo adicional em certo calendário, mas não vira obrigação universal de compra. Cuidado depende do ciclo que efetivamente pagará o bônus, não de um corte fixo descontextualizado no dia 28. |
| `CAPHARV`: teto dos animais | Projeção da produção noturna eleva urgência de colheita antes de exceder o teto. | O bônus antigo é pago antes do crédito de cuidado do dia; o modelo respeita essa ordem e o limite por espécie. |
| `SHEDROOM` / `OVERFLOW` | Nas últimas duas horas, vendas e compras consideram galpão mais carga prevista após ações, buscando liberar espaço antes do depósito automático. | Produto carregado só pode ser vendido após depósito. O cálculo evita contar novamente trigo já depositado e pode liberar fertilizante reservado sob pressão. |
| `ORDERPRI2` | Ordena vendas pela exposição à oferta adversária inferida e prioriza alimentação urgente; libera espaço por vendas quando uma compra de ração exigiria capacidade. | Não lê a fila adversária. Posições anteriores podem preceder posições posteriores; ordens simultâneas continuam obedecendo à cotação pareada do motor. |
| Conservação de massa e estoque rival | Resíduos de estoque corrigidos por demanda/ordens próprias e mudanças de rendimento visível alimentam uma estimativa de estoque e risco. | Compras, piso, transporte e ordens frustradas impedem reconstrução exata. Estoque inferido não é inventário privado observado. |
| Fingerprinting determinístico | Substituído por perfil heurístico público: passivo, pecuário, concentrado ou diversificado. | Ordens iniciais, sementes e galpão privados não são observáveis. Não se identifica exatamente um notebook nem se conhece sua trajetória futura. |
| Predatory price crashing | Antecipação e prioridade de venda de produtos próprios disponíveis diante de pressão estimada. | Não há acumulação obrigatória para sabotagem, nem certeza de colheita/venda do rival ou garantia de colapso do preço. A prioridade econômica continua ser liquidez e retorno próprios. |
| Wheat squeeze / starvation | Somente a opção experimental `tactical_wheat_reserve`, desativada, aumenta de forma limitada a reserva própria de ração. | Mercado de trigo não funciona como estoque finito bloqueável por compras. Não se promete impedir a alimentação adversária; caixa e capacidade próprios continuam limitando a reserva. |
| Contextual bandit bayesiano | Thompson sampling linear gaussiano disponível, com contexto público, atualização diária e permanência mínima do braço. **Desativado por padrão.** | Recompensa atrasada é confundida por investimentos anteriores. Não é probabilidade calibrada de vitória nem aprendizagem persistida entre jogos. A liquidação final tem precedência. |
| Bucket brigade / `COURIER` | Zonas, persistência de tarefa e urgência de entrega para a unidade que já carrega os produtos. | Não existe transferência direta de inventário entre trabalhadores no motor. Não foi inventada uma ação de repasse nem uma esteira logística impossível. |
| `CARROT2` e culturas adaptativas | As cinco culturas já competem no portfólio, com restrições de recursos e execução de plantio/rega. | Não se importou um overlay de fita. A avaliação econômica de coortes não é uma simulação física perfeita de cada cultura futura. |
| Segurança da abertura | Reservas operacionais, compras financiadas com caixa já disponível, alimentação antes de investimento discricionário e contabilidade após ações. | Não se repetiu um padrão especulativo de compra/venda inicial dependente de preços ou rota do notebook. |
| Loader e wrappers | Entrada própria `main.py` e validação do arquivo extraído em processo isolado. | A arquitetura não empilha wrappers redefinindo o mesmo símbolo de agente; o truque de reinserção do loader citado no notebook não foi copiado sem necessidade. |

Aplicações de fertilizante reservam unidade do insumo, célula, percurso e rega subsequente quando necessária. O manejo preserva a trava do primeiro plantio: `PLANT` compromete a mesma unidade com `WATER` no turno seguinte do mesmo dia. Compra, instalação, alimentação e transporte de animais continuam acompanhando o estado observado, incluindo limpeza de ervas e tentativas retomadas.

## Avaliação e diferenças entre hipótese e evidência

A suíte final desta etapa registra **160 testes aprovados**, incluindo oráculos de força bruta para pequenos problemas inteiros, limites e gaps do B&B, métodos numéricos, fertilização, prioridade de mercado, proteção noturna, treino e contratos do motor. O registro está em [test-results_0.1.2.txt](../relatorios/test-results_0.1.2.txt). Isso verifica os comportamentos cobertos pelos testes; não mede, por si só, força competitiva.

Os resultados econômicos, tempos medidos e validação do pacote final são apresentados separadamente no [índice de resultados da 0.1.2](../README.md#resultados-da-012). São usados confrontos reativos completos com ambos os lados e referências congeladas. Replays servem à auditoria e ao treino, não a oponentes que seguem ações impossíveis após uma mudança de mercado.

Bradley–Terry permanece uma medida local regularizada dos confrontos efetivamente realizados. Amostras pequenas, seeds compartilhadas, oponentes sintéticos relacionados e componentes desconectados limitam a leitura. As pontuações e narrativas dos notebooks de referência não foram transformadas em garantia de Top 1, teto teórico de rating ou previsão de posição no Kaggle.

A referência preservada da 0.1.1 é `dist/submission-0.1.1.tar.gz`. O antigo `benchmark_0.1.1_comparison.json` pertence a um **snapshot intermediário anterior ao pacote final**, e não deve ser usado como medida do artefato congelado sem essa distinção. A proveniência está esclarecida no [registro histórico 0.1.1](../relatorios/RESULTADOS_0.1.1.md).

Para reproduzir comandos, variantes e treinamento, consulte o [README](../../README.md). A descrição detalhada dos contratos internos está em [ARCHITECTURE.md](../tecnica/ARCHITECTURE.md).
