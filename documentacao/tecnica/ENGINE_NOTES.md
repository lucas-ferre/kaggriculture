# Motor fixado e contratos verificados

## Proveniência

- Repositório: https://github.com/Kaggle/kaggle-environments
- Commit do ambiente: 28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c
- Arquivos originais: vendor/kaggriculture/.
- Licença original: vendor/kaggriculture/LICENSE.
- Framework instalado para avaliação: kaggle-environments==1.32.7.

O benchmark registra explicitamente esse motor com o nome local kaggriculture_pinned. Ele usa a infraestrutura oficial, sem implementar um simulador substituto. A cópia do motor não foi alterada e não entra no pacote do agente. O hash é registrado em cada relatório. Uma atualização no Kaggle exige nova auditoria antes de assumir equivalência.

## Contratos confirmados por código e testes de regressão

| Questão | Comportamento do snapshot |
| --- | --- |
| PLACE em galpão cheio | Transfere somente o que cabe; mantém excedente no inventário da unidade. |
| DROP e depósito automático | Descartam excedente da capacidade. |
| Fonte de SELL | Apenas private.shed. |
| Depósito e venda no mesmo turno | Possível: ações das unidades precedem o mercado. |
| Depósito automático de fim de dia | Ocorre depois do mercado; produto não pode ser vendido retroativamente. |
| Compra e plantio no mesmo turno | A compra chega depois de PLANT; precisa de decisão posterior. |
| Contratação | Mão contratada recebe ações a partir do turno seguinte. |
| Sementes insuficientes | Todas as requisições PLANT daquela cultura no turno são bloqueadas. |
| Plantio sem rega | Começa com contador 1 e morre na primeira virada se não for regado. |
| Melão | Pode acumular unidades antes da maturação, mas HARVEST é bloqueado antes da idade 10. |
| Tomate e morango | Quatro eventos de produção. Fertilização muda unidades por evento, não prolonga o calendário; capacidade simultânea é quatro unidades. |
| Construções | BUILD_COOP e BUILD_PASTURE não descontam moedas neste snapshot; consomem ação. |
| FEED | Consome trigo do inventário da unidade, não diretamente do galpão. |
| Turno final | Com episodeSteps=720: decisões 0–718, estado final step=719. Não ocorre a virada que seria processada na ação 719. |
| Reward | Saldo monetário final. |
| Seed aleatória | Guardada no ambiente para replay e removida da configuração entregue ao agente. |

A observação e a configuração são suficientes para aplicar as regras acima. O agente não consulta env.info ou a seed para tomar decisões. O primeiro guia documental foi escrito antes desta auditoria; os pontos resolvidos aqui têm a versão explicitada.

## Dependências de desenvolvimento

O agente submetido usa somente stdlib. Para rodar o framework e os testes de motor, a instalação mínima deste workspace contém kaggle-environments 1.32.7 sem o conjunto completo de dependências de outros jogos, mais requests/jsonschema e dependências transitivas listadas em requirements-core.txt.

Alguns ambientes do catálogo emitem mensagens de dependências ausentes no import global. O benchmark captura essas mensagens nos metadados e carrega o ambiente Kaggriculture fixado. A suíte de contratos e os episódios completos verificam esse caminho específico; a instalação mínima não habilita todos os jogos do pacote.

No Windows, a distribuição completa contém caminhos muito longos. Neste workspace foi necessário instalar o wheel com prefixo de caminho estendido para o diretório site-packages. Caso a instalação convencional falhe por caminho, usar uma pasta curta para a venv ou habilitar suporte a caminhos longos conforme a política da máquina. Não é necessário instalar GPU, Torch ou um solver comercial para executar este agente.

## Escopo da verificação

Os testes de motor cobrem os contratos que orientam a execução, não todas as ações possíveis do jogo. Os testes de mercado comparam a fórmula ao código fixado em ambos os lados da curva e com overrides. Os testes do planejador e do executor verificam decisões nas bordas do horizonte, recursos compartilhados e isolamento do estado.
