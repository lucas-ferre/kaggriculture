# Atualização 0.1.1 — decisões de implementação

A atualização aplica o diagnóstico de [ATUALIZACOES_INICIAIS_0.1.1.md](../referencias/ATUALIZACOES_INICIAIS_0.1.1.md) ao agente executável. O documento de análise original e os resultados históricos da 0.1.0 foram preservados. O motor de desenvolvimento continua no commit `28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c`, sem alterações.

## O que mudou

| Proposta do documento | Implementação na 0.1.1 | Limite da implementação |
| --- | --- | --- |
| Pecuária, expansão e capital produtivo | Aquisição gradual de vacas, ovelhas e, quando configurado, gansos; construção, instalação, alimentação, cuidado, colheita e coleta/venda de fertilizante. Compra de terra condicionada a caixa, mão de obra e horizonte. | Padrão permite até 8 vacas, 4 ovelhas e 2 expansões; os limites não obrigam compras sem utilidade econômica. Fertilizante é monetizado; não há política de aplicação nas plantas. |
| Portfólio livre | As cinco culturas competem por terra, caixa, sementes, mão de obra e capacidade de lote. | Plantas existentes são preservadas. Custos de execução e produtividade são aproximados. |
| Simplex nativo | Solver contínuo de duas fases, estados explícitos, normalização, tranches marginais e conversão para quantidades inteiras viáveis. | Simplex resolve LP, não MILP. O arredondamento não certifica ótimo inteiro; pecuária e culturas são planejadas sequencialmente. |
| Corrigir receita agregada | Lotes cronológicos; tomate e morango têm eventos separados. As vendas próprias anteriores continuam afetando os estoques projetados. | Coortes de plantio, demanda e oferta adversária são aproximações. Interações no piso de preço entre oferta externa e vendas próprias não são um replay exato do futuro. |
| Risco por quantil | Quantil empírico de receita dos cenários, configurável por `risk_quantile`. | Não é regressão quantílica treinada. Faltam dados de treino e validação temporal para publicar coeficientes calibrados. |
| Adversário vendendo em pulsos | Maturação visível produz eventos discretos; cenários variam realização, atraso e oferta não observada. | Não há acesso ao estoque privado adversário nem certeza sobre suas ações futuras. |
| Venda antecipada | Produtos premium já disponíveis podem ser vendidos antes de um possível pulso de oferta, se não houver consumo da cidade naquele turno. | As ordens simultâneas são cotadas em lockstep no motor. Vender um turno antes não concede prioridade dentro do mesmo turno. |
| Regar sementes novas com segurança | `PLANT` reserva `WATER` para o mesmo trabalhador no turno imediatamente seguinte; plantio é bloqueado quando esse turno não cabe no mesmo dia. | A garantia corresponde à continuidade normal das chamadas e ações do próprio agente. Reinício recupera urgência pela observação, mas não é uma simulação matemática de todas as ações futuras. |
| Organização espacial | Núcleo pecuário próximo ao galpão, preferências de zonas e persistência de destinos. | Atribuição gulosa/Húngaro foi mantida. Não foi substituída por campos de potencial, que por si só não garantem rotas sem oscilação ou melhor desempenho. |
| Ensembles dinâmicos | Seleção determinística entre especialistas de margem, caixa, contrarian e liquidação conforme observação. | Não é um contextual bandit treinado; não há votação de ações incompatíveis entre múltiplos executores. |
| Arena multimodelo | Baseline original congelado, starter, autojogo e perfis sintéticos; duas posições por seed; métricas econômicas, físicas e de tempo; Bradley–Terry regularizado. | Perfis sintéticos reutilizam a implementação. Rating local não reproduz nem prevê o ranking oficial. |

## Correções encontradas nas partidas e na revisão

- A instalação de novos animais passou a reservar previamente o núcleo próximo ao galpão, evitando que culturas deslocassem a pecuária para bordas distantes.
- Um trabalhador pode buscar ração para um animal mesmo quando outro trabalhador distante já carrega trigo; o livro de reservas compartilhado continua impedindo retirar mais que o disponível.
- Compra e retenção de ração usam o mesmo alvo, com reposição por faixa de estoque. Trigo depositado deixa de contar simultaneamente como estoque carregado.
- O planejador reserva a área do mesmo núcleo pecuário que o executor protege, reduzindo compra de sementes para parcelas indisponíveis.
- Replanejamento considera o lote projetado das culturas já existentes ao liberar capacidade para novas plantações.
- O solver trata coeficientes muito pequenos por normalização e distingue problemas ilimitados, inviáveis e falhas numéricas.

## Evidência e reprodução

Os resultados consolidados, comandos, seeds, limites da amostra e arquivos de evidência ficam em [RESULTADOS_0.1.1.md](../relatorios/RESULTADOS_0.1.1.md). As verificações incluem suíte de regressões, episódios completos no motor fixado e execução do pacote extraído em um processo Python isolado.

O arquivo `dist/submission-0.1.0.tar.gz` preserva o runtime original. A opção `baseline_010` o importa em namespace separado, sem substituir os módulos atuais. `legacy_policy` é outra comparação: parâmetros antigos executados sobre as correções atuais, e não o código original.

## Próximos experimentos fundamentados em dados

1. Coletar replays e previsões prospectivas, separar treino/validação por episódio e tempo, medir cobertura dos quantis e erro de receita antes de habilitar regressão quantílica ajustada.
2. Comparar a seleção heurística de especialistas com um bandit usando o mesmo orçamento de partidas e adversários independentes.
3. Avaliar aplicação de fertilizante pelo ganho marginal líquido, comparando com sua venda e incluindo deslocamento e mão de obra.
4. Usar adversários competitivos independentes para medir robustez. O notebook Boatlee citado na análise não foi incorporado ao pacote nem reavaliado nesta entrega; suas pontuações permanecem referências históricas do documento original.

A meta de mais de 140 mil moedas não é condição garantida pelo solver nem previsão de ranking. O saldo depende da seed, lado, adversário, demanda e capacidade efetiva de executar o plano.
