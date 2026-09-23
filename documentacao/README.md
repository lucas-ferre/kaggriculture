# Documentação de apoio

Esta pasta reúne os materiais de desenvolvimento, as referências históricas e os relatórios do projeto. **Nenhum arquivo de `documentacao/` integra a submissão.**

O [empacotador](../scripts/package_submission.py) seleciona apenas `main.py`, os módulos e dados locais de `kaggriculture_agent/`, `LICENSE-APACHE-2.0.txt` e `THIRD_PARTY_NOTICE.txt`. O arquivo a enviar é [dist/submission.tar.gz](../dist/submission.tar.gz). Os comandos de desenvolvimento devem ser executados a partir da raiz do projeto.

## Organização

| Pasta | Conteúdo |
| --- | --- |
| `tecnica/` | Arquitetura atual, contratos do motor e preparação do ambiente. |
| `versoes/` | Notas das versões e registro de prontidão para a competição. |
| `referencias/` | Propostas originais, análises de concorrentes, guia inicial e manual de submissão. |
| `relatorios/` | Resultados históricos em texto e registros das suítes de testes. |

Os arquivos JSON de experimentos e proveniência continuam em [outputs/](../outputs/), e a referência de reprodução permanece em [reports/](../reports/). Eles também ficam fora da submissão.

## Documentação técnica

- [Arquitetura 0.1.2](tecnica/ARCHITECTURE.md).
- [Contratos do motor e ambiente de desenvolvimento](tecnica/ENGINE_NOTES.md).
- [Configuração do Kaggle CLI, teste de acesso e envio manual](tecnica/KAGGLE.md).
- [Notas da versão 0.1.2](versoes/RELEASE_0.1.2.md).
- [Notas da versão 0.1.1](versoes/RELEASE_0.1.1.md).

## Resultados da 0.1.2

Os resultados desta versão estão nos arquivos abaixo. O antigo caminho `outputs/RESULTADOS_0.1.2.md` não continha um arquivo; as referências foram substituídas por este índice das evidências existentes.

- [Comparação da 0.1.2 e da 0.1.1 contra starter](../outputs/benchmark_0.1.2_comparison.json).
- [Arena com a 0.1.1, perfis locais e autojogo](../outputs/arena_0.1.2.json).
- [Comparações entre mecanismos](../outputs/ablations_0.1.2.json) e [controle com a política padrão](../outputs/ablation_control_0.1.2.json).
- [Calibração dos modelos quantílicos](../outputs/quantile_calibration_0.1.2.json).
- [Registro de 160 testes aprovados](relatorios/test-results_0.1.2.txt).
- [Validação do pacote extraído](../outputs/entrypoint_validation_0.1.2.json).
- [Nova validação antes do envio](../outputs/pre_submission_validation_0.1.2.json) e [verificação de autenticação do Kaggle](../outputs/kaggle_auth_check.json).
- [Manifesto do artefato](../outputs/manifest_0.1.2.json) e [conferência do motor público](../outputs/upstream_check_0.1.2.json).

Esses registros são avaliações locais. Amostras pequenas e adversários relacionados limitam a generalização dos resultados e não demonstram uma colocação no ranking oficial.

## Histórico e referências

- [Resultados da 0.1.0](relatorios/RESULTADOS.md), [registro da 0.1.1](relatorios/RESULTADOS_0.1.1.md) e [registro inicial de testes](relatorios/test-results.txt).
- Propostas originais: [0.1.1](referencias/ATUALIZACOES_INICIAIS_0.1.1.md) e [0.1.2](referencias/ATUALIZACOES_INICIAIS_0.1.2.md).
- [Guia básico](<referencias/Guia Basico.txt>) e [manual de submissão](referencias/MANUAL_SUBMISSAO.md.pdf).

As propostas e os guias iniciais são referências históricas. Para o comportamento implementado, consulte a documentação técnica e as notas da versão.
