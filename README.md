# Kaggriculture — agente adaptativo 0.1.3

A versão 0.1.3 acrescenta histórico persistente de partidas, calibração de probabilidades por Regressão Linear Múltipla (MRLM), expansão por capacidade produtiva próxima ao galpão e limite de seis ovelhas. Tomate e morango recebem preferência moderada quando rentáveis e podem iniciar ciclos parciais com pelo menos dois eventos de produção antes do prazo final. Alimentação de animais existentes pode utilizar a reserva de caixa; novos investimentos precisam preservar o funcionamento da fazenda.

O código executado durante a partida utiliza somente a biblioteca padrão de Python. O motor Kaggle, replays e treinamento offline pertencem ao ambiente de desenvolvimento. Consulte [as notas da versão atual](documentacao/versoes/RELEASE_0.1.3.md), [os registros históricos da 0.1.2](documentacao/README.md#resultados-da-012) e [a arquitetura anterior](documentacao/tecnica/ARCHITECTURE.md). Resultados locais não reproduzem o rating oficial do Kaggle.

## Histórico e calibração MRLM

Cada execução do benchmark grava os resultados em `outputs/match_history.jsonl` e, por padrão, os replays completos em `outputs/replays/`. O histórico registra seed, lado, políticas, versão/hash do código e do motor, resultado, expansões, culturas e animais. Repetir uma seed com outra política preserva o replay anterior; a importação de resultados idênticos não duplica o registro. Use `--history` e `--replays` para escolher outros caminhos; `--no-replays` mantém somente o resumo, que não basta para treinar a MRLM. Não rode dois escritores simultâneos no mesmo histórico.

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark --variants adaptive no_mrlm --seeds 83 97 131 --opponents baseline_012 --output outputs/comparison_0.1.3.json
.\.venv\Scripts\python.exe -m scripts.match_history --import-report outputs/quantile_training_games.json outputs/quantile_validation_games.json
```

O treinamento usa três grupos de partidas com seeds diferentes: ajuste dos coeficientes, calibração dos resíduos e validação. A MRLM estima o preço futuro usando atributos públicos; a distribuição dos resíduos de calibração transforma a estimativa em probabilidades de alta, estabilidade e queda. Esses números descrevem preços, não a chance de vencer a partida. A validação mede Brier multiclasses e calibração por faixas de probabilidade; modelos reprovados, horizontes incompatíveis ou atributos fora do domínio conservam a previsão empírica. Receita de lotes e impacto de mercado continuam calculados pelo modelo de cenários/quadratura.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mrlm --train-replays tmp/quantile_train --calibration-replays tmp/mrlm_calibration --validation-replays tmp/quantile_validation --output outputs/mrlm_report_new.json --model-output outputs/mrlm_candidate_new.json
```

Também é possível selecionar os três conjuntos pelo histórico com `--history outputs/match_history.jsonl --train-seeds 101 103 --calibration-seeds 151 157 --validation-seeds 211 223`, no lugar das opções de replays. O treinador verifica os hashes e rejeita interseção de seeds/episódios. A gravação do candidato não altera automaticamente o modelo embarcado nem os coeficientes quantílicos. Partidas oficiais podem entrar no treinamento quando seus replays completos estiverem disponíveis localmente; o intervalo de rating informado pelo usuário não fornece observações suficientes para ajustar uma regressão.

## Executar no ambiente preparado

No PowerShell, a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.benchmark --variants adaptive baseline_011 --seeds 11 29 47 --opponents starter baseline_011 --output outputs/benchmark_0.1.2.json
```

Para comparar mecanismos contra perfis reativos locais, com as mesmas seeds e os dois lados:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark --variants adaptive baseline_011 --seeds 11 29 47 --opponents passive crop_flood livestock expansive selfplay --output outputs/arena_0.1.2.json
.\.venv\Scripts\python.exe -m scripts.benchmark --variants adaptive scenarios no_fertilizer simplex bandit --seeds 11 29 47 --opponents baseline_011 --output outputs/ablations_0.1.2.json
```

Os comandos especificam `baseline_011` para comparar com a versão anterior preservada. Ajuste opcional de parâmetros, com seeds de validação separadas:

```powershell
.\.venv\Scripts\python.exe -m scripts.tune --trials 3 --train-seeds 101 --validation-seeds 211 --opponent selfplay --output outputs/tuning_0.1.2.json
```

O tuner compara busca aleatória e hill climbing com orçamento igual. Registra a configuração escolhida e a validação sem modificar a política padrão automaticamente.

Empacotamento e validação do artefato extraído em processo Python isolado:

```powershell
.\.venv\Scripts\python.exe -m scripts.package_submission
.\.venv\Scripts\python.exe -m scripts.validate_submission --output outputs/entrypoint_validation_0.1.2.json
```

O empacotador gera `dist/submission.tar.gz`, com entrada `main.py`, e não envia o arquivo ao Kaggle. Os arquivos `dist/submission-0.1.1.tar.gz` e `dist/submission-0.1.0.tar.gz` permanecem como referências congeladas, carregadas em namespaces isolados por `baseline_011` e `baseline_010`.

## Estrutura

Para autenticar o CLI com o `kaggle.json` local, testar o acesso e consultar o comando de envio manual, veja [Configuração do Kaggle](documentacao/tecnica/KAGGLE.md). O atalho `scripts/kaggle.ps1` utiliza o CLI instalado em `.venv` e a credencial da raiz do projeto.

A documentação de apoio está reunida em [documentacao/](documentacao/README.md), com referências históricas, notas de versão, arquitetura e relatórios. Essa pasta fica fora da submissão: o empacotador inclui somente `main.py`, os arquivos de runtime de `kaggriculture_agent/` e as duas licenças/atribuições da raiz.

| Caminho | Responsabilidade |
| --- | --- |
| `main.py` | Entrada `agent(observation, configuration)` e instâncias separadas por jogador. |
| `kaggriculture_agent/market.py` | Curva oficial, lotes inteiros, histórico, pulsos públicos, demanda futura e receitas por lotes cronológicos. |
| `kaggriculture_agent/numerics.py` | Nós/pesos da quadratura normal de cinco pontos, quantil ponderado e tendência causal de sete amostras. |
| `kaggriculture_agent/quantile.py` / `mrlm.py` | Atributos públicos, quantis de preço e probabilidades MRLM aprovadas e compatíveis. |
| `kaggriculture_agent/economy.py` | Investimento gradual em animais, reservas de recursos e prioridade das ordens de mercado. |
| `kaggriculture_agent/planner.py` | Portfólio, comparação de equipes por custo Fibonacci e diagnóstico da otimização inteira. |
| `kaggriculture_agent/optimization.py` | Simplex de duas fases, branch-and-bound limitado, enumeração/beam legados e atribuição gulosa/Húngaro. |
| `kaggriculture_agent/agronomy.py` | Calendários de produção, bônus de cuidado, tetos de estoque e valor marginal do fertilizante. |
| `kaggriculture_agent/intelligence.py` | Perfil adversário público e controlador contextual experimental. |
| `kaggriculture_agent/scheduler.py` | Zonas, continuidade de tarefas, rega reservada, fertilização, manejo animal e transporte. |
| `kaggriculture_agent/agent.py` | Coordenação, contratação, compras, reservas, proteção de meia-noite e vendas. |
| `scripts/benchmark.py` / `scripts/arena.py` | Partidas no motor fixado, auditoria e ratings Bradley–Terry locais. |
| `scripts/train_quantile.py` | Treinamento pinball/L2 e avaliação em replays separados por episódio e seed. |
| `scripts/train_mrlm.py` / `scripts/match_history.py` | Ajuste MRLM/L2, calibração/validação separadas e histórico persistente com hashes de replays. |
| `scripts/tune.py` | Ajuste offline de parâmetros com validação separada. |
| `tests/` | Contratos do motor e regressões numéricas, econômicas, de execução, treinamento e empacotamento. |
| `documentacao/` | Documentação de apoio organizada por assunto, excluída da submissão. |
| `vendor/kaggriculture/` | Snapshot não modificado do motor oficial, fora da submissão. |

Os contratos auditados estão em [ENGINE_NOTES.md](documentacao/tecnica/ENGINE_NOTES.md). O [relatório histórico da 0.1.0](documentacao/relatorios/RESULTADOS.md) descreve a **0.1.0**; o [registro histórico da 0.1.1](documentacao/relatorios/RESULTADOS_0.1.1.md) distingue a validação do pacote final dos diagnósticos intermediários. `benchmark_0.1.1_comparison.json` é anterior ao artefato final preservado e não mede o `baseline_011` final. Resultados de versões diferentes permanecem separados.

## Política padrão e controles experimentais

| Parâmetro | Padrão e efeito |
| --- | --- |
| `strategy` / `optimizer` | `adaptive` / `branch_bound`: portfólio das cinco culturas com solução inteira sob orçamento. |
| `integer_node_limit` | 64 nós de LP compartilhados entre as opções de equipe do plano agrícola; o investimento animal tem sua própria chamada limitada. |
| `market_method` | `quadrature`: cinco trajetórias com pesos de Gauss–Hermite sob aproximação normal. `scenarios` mantém a alternativa de 32 cenários. |
| `target_workers` / `max_plots` | Tetos de oito unidades no total e 40 parcelas; o plano escolhe a equipe economicamente justificável. |
| `cash_reserve` | 300 moedas, além das reservas operacionais aplicáveis. |
| `risk_aversion` / `risk_quantile` | 0,35 / 0,25. |
| `enable_livestock` / `enable_expansion` | Ativos, com limites de oito vacas, seis ovelhas, nenhum ganso e três expansões; compras dependem de retorno e capacidade. |
| `ongoing_preference` / `min_recurrent_pulses` | Preferência de 0,10 sobre margem positiva de tomate/morango; ao menos dois eventos de produção por novo ciclo. |
| `enable_mrlm` | Permite probabilidades aprendidas somente quando o modelo embarcado é aprovado e compatível; `no_mrlm` desativa na comparação. |
| `enable_fertilizer` | Ativo: aplicação exige ganho incremental positivo após custo de oportunidade e trabalho. |
| `enable_opponent_model` | Ativo: perfil e estoque inferidos a partir de sinais públicos, com incerteza. |
| `enable_contextual_bandit` | `False`: Thompson sampling contextual disponível como experimento. |
| `tactical_wheat_reserve` | `False`: reserva adicional limitada de ração própria, experimental. |

Animais e culturas são alocados sequencialmente. Quando a pecuária começa, preserva-se espaço próximo ao galpão para o núcleo eventual de até 14 animais na configuração padrão. O plano agrícola compara capacidade e remuneração de equipes; contratações imediatas consomem caixa antes da compra de sementes. O motor permite atravessar terrenos bloqueados e determina a ordem das compras de terreno: NE, SW, SE. A política compra espaço produtivo com retorno previsto, em vez de comprar apenas porque um funcionário passou pelo local.

Fertilizante pode ser aplicado quando a projeção de unidades adicionais, respeitando os tetos da planta e as ações necessárias, supera sua venda. Melão já próximo do teto pode não ganhar nada com a aplicação. A rotina protege a rega, evita consumo duplicado do insumo e libera excedentes para venda.

## Variantes e arena

| Variante | Comparação |
| --- | --- |
| `adaptive` | Política padrão 0.1.3. |
| `baseline_012` / `baseline_011` / `baseline_010` | Código e dados congelados das respectivas versões, preservados nos arquivos de submissão. |
| `no_mrlm` | Política atual com probabilidades empíricas, para isolar o efeito do modelo aprendido. |
| `scenarios` | Troca a quadratura pelos 32 cenários. |
| `no_fertilizer` | Desabilita aplicação agronômica de fertilizante. |
| `simplex` | Usa relaxamento contínuo e reparo viável no lugar do branch-and-bound. |
| `bandit` | Ativa o controlador contextual experimental. |
| `legacy_policy` / `dual_forecast` | Parâmetros antigos de duplo ciclo sobre o runtime atual. |
| `dual_spot` / `fast_only` / `long_any` | Ablações legadas de previsão, conjunto de culturas e admissão de melão. |
| `hungarian` / `beam` | Política legada com outra atribuição de tarefas ou busca do par de culturas. |

Adversários: `starter`, `random`, `pass`, `selfplay`, `baseline_011`, `baseline_010`, `legacy_policy`, `passive`, `crop_flood`, `livestock` e `expansive`. Os perfis sintéticos compartilham este runtime; `selfplay` usa outra instância da política atual. O agente `random` possui aleatoriedade própria, limitando a reprodução somente pela seed da partida.

`--replays outputs/replays_0.1.2` salva partidas completas. A auditoria distingue ações solicitadas, efeitos observados e ações sem efeito; `--skip-action-audit` omite essa reconstrução posterior. Os relatórios incluem configurações, hashes do runtime/motor, status, saldos, margens, tempos e uso de recursos.

Bradley–Terry resume apenas confrontos realizados, com regularização e meio ponto por empate. Partidas inválidas são excluídas. A dispersão depende do prior e de aproximação de Laplace. Seeds compartilhadas, amostra pequena, famílias relacionadas e componentes desconectados limitam a interpretação. O rating local não reproduz a avaliação oficial nem estima posição no leaderboard.

## Treinamento quantílico offline

O pipeline inicial usa oito replays locais completos: quatro para ajuste e quatro para validação, com seeds separadas. Esses jogos geram exemplos supervisionados; não são reproduzidos como oponentes congelados. Para reproduzir o conjunto inicial e o ajuste:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark --variants baseline_011 --seeds 101 103 --opponents baseline_010 --replays tmp/quantile_train --output outputs/quantile_training_games.json
.\.venv\Scripts\python.exe -m scripts.benchmark --variants baseline_011 --seeds 211 223 --opponents starter --replays tmp/quantile_validation --output outputs/quantile_validation_games.json
.\.venv\Scripts\python.exe -m scripts.train_quantile --train-replays tmp/quantile_train --validation-replays tmp/quantile_validation --output outputs/quantile_training_report.json --model-output outputs/quantile_model_candidate.json
```

O alvo é o preço público futuro em horizonte fixo, com perda pinball e regularização L2. A padronização usa somente treino. Cada produto precisa passar critérios de quantidade de exemplos, melhoria da perda e cobertura na validação. Apenas coeficientes aprovados e compatíveis podem ser incorporados ao arquivo local `kaggriculture_agent/quantile_coefficients.json`; o comando acima grava um candidato separado para revisão.

A aprovação é um filtro exploratório de implantação sobre amostra pequena, não uma prova de ganho competitivo nem um teste final intocado. A inferência exige produto, horizonte, quantil e faixa de atributos compatíveis. Fora dessas condições, o agente utiliza sua distribuição empírica. Quantis de **receita de vários lotes** permanecem calculados sobre as trajetórias de mercado; não são substituídos por um quantil de preço isolado. O modelo embarcado inicial contém somente `WOOL` e `FERTILIZER`, aprovados no filtro local para horizonte de 24 turnos e quantil 0,25. A aprovação não demonstra ganho competitivo; métricas e limitações estão no [relatório de calibração](outputs/quantile_calibration_0.1.2.json) e nas [notas da versão](documentacao/versoes/RELEASE_0.1.2.md).

## Instalação e alcance

O agente requer Python 3.11+. O ambiente preparado usa Python 3.12 e `kaggle-environments==1.32.7`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m pip install --no-deps kaggle-environments==1.32.7
```

O benchmark registra o motor em `vendor/kaggriculture/`. Avisos sobre dependências de outros jogos ficam nos metadados. Detalhes do ambiente mínimo e dos caminhos longos do Windows estão em [ENGINE_NOTES.md](documentacao/tecnica/ENGINE_NOTES.md).

Branch-and-bound mantém incumbente, limite superior e gap; só certifica o modelo inteiro quando a árvore fecha. Não existe garantia de fechar todos os problemas em 3–5 ms. A quadratura integra uma aproximação normal de cinco pontos, sem equivalência garantida a milhares de simulações. A escolha do lote usa bisseção inteira na cotação oficial arredondada e com piso, evitando pressupor uma derivada útil para Newton. O agente não observa ordens privadas nem identifica exatamente um bot por assinatura. Seu transporte prioriza o trabalhador que já carrega a mercadoria; o motor não oferece transferência direta entre unidades.
