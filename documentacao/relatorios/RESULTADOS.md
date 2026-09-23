# Resultados da primeira arquitetura

Versão 0.1.0, avaliação local em 19/09/2026. Motor oficial fixado no commit 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c; framework 1.32.7. Parâmetros e evidências completas estão nos JSONs de [outputs/](../../outputs/); hashes do código e pacote em [manifest.json](../../outputs/manifest.json).

## Comparação das variantes

36 partidas completas contra o starter oficial: seis variantes, seeds 11/29/47, ambas as posições. Todas terminaram válidas; cada variante venceu as seis partidas. Esse adversário simples serve para comparação local, não estima força no leaderboard. Seeds espelhadas não são observações independentes.

| Variante | Partidas | Saldo final médio | Maior decisão local (ms) |
| --- | --- | --- | --- |
| Duplo recorrente + previsão | 6 | 21063.17 | 63.70 |
| Duplo recorrente + mercado atual | 6 | 20105.33 | 26.27 |
| Somente rápido | 6 | 13168.67 | 34.18 |
| Duplo + previsão + Húngaro | 6 | 21015.67 | 46.17 |
| Duplo + previsão + beam search | 6 | 21063.17 | 46.69 |
| Duplo com melão permitido | 6 | 40696.50 | 54.18 |

Nos cenários observados, o saldo médio do modo recorrente com previsão foi 4.76% maior que o da variante sem projeção temporal, mantendo o impacto da própria venda em ambas. Foi 59.95% maior que o do ciclo apenas rápido. Essa diferença é descritiva, baseada em três seeds e um adversário simples; não é significância estatística nem garantia de generalização.

Húngaro ficou próximo do método guloso, sem ganho claro. Beam search encontrou as mesmas decisões econômicas/resultados nesta amostra. Permitir melão aumentou bastante o saldo contra starter, mas muda a natureza do ciclo longo: melão só permite uma colheita. O padrão mantém tomate/morango para corresponder ao cultivo recorrente solicitado; long_any permite comparar a alternativa de maior retorno local.

## Busca de parâmetros

Busca aleatória e hill climbing receberam três candidatos cada, incluindo o padrão, com seed de ajuste 101 e ambos os lados contra uma instância independente do agente padrão. Depois foram avaliados em seed 211, não usada na seleção. São 18 partidas ao todo, incluindo avaliações do padrão, todas válidas.

A busca aleatória reteve o padrão. O hill climbing selecionou fast_share=0.7, com ganho médio de 4 moedas no ajuste e margem média de 170 moedas na validação contra o agente padrão. São amostras pequenas: a configuração de produção não foi alterada automaticamente. As demais configurações e partidas estão em [tuning.json](../../outputs/tuning.json).

## Verificações de execução

- 50 testes passaram: regras reais do motor, preço, probabilidades, planejamento, atribuição, ações e pacote isolado.
- Pacote determinístico com main.py na raiz e somente módulos runtime, licença e atribuição; cerca de 23 KiB.
- O arquivo empacotado foi extraído e executado em Python -I, sem carregar o código da pasta de desenvolvimento. Autojogo completo: 720 estados, dois status DONE, sem erros.
- Nenhum acesso à rede ou serviço externo é necessário durante uma decisão.

O limite de 1 segundo por decisão consta no motor; os tempos da tabela são medições locais, não certificação do hardware do Kaggle. O benchmark registra tempos por partida. Probabilidades ainda são cenários heurísticos; calibração por Brier score/cobertura e comparação contra adversários fortes continuam pendentes.

## Reproduzir

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
    .\.venv\Scripts\python.exe -m scripts.benchmark --variants dual_forecast dual_spot fast_only hungarian beam long_any --seeds 11 29 47 --opponents starter --output outputs/benchmark.json
    .\.venv\Scripts\python.exe -m scripts.tune --trials 3 --train-seeds 101 --validation-seeds 211 --opponent selfplay --output outputs/tuning.json
    .\.venv\Scripts\python.exe -m scripts.package_submission
    .\.venv\Scripts\python.exe -m scripts.validate_submission

Os tempos variam a cada execução. O starter é determinístico; o random oficial não fixa seu gerador por episódio e, por isso, não foi usado nesta comparação reproduzível.
