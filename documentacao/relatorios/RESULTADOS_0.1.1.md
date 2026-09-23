# Registro histórico — versão 0.1.1

Este arquivo separa a validação do **pacote final preservado da 0.1.1** dos diagnósticos produzidos durante seu desenvolvimento. Não representa os resultados da 0.1.2.

## Artefato final preservado

A referência é [`dist/submission-0.1.1.tar.gz`](../../dist/submission-0.1.1.tar.gz), com SHA-256:

```text
87f7a8f80cce1645fb948a5443a83af9bf1ab0ea45d572426c818a8b9eba7982
```

O hash do arquivo preservado coincide com `archive_sha256` de [entrypoint_validation_0.1.1.json](../../outputs/entrypoint_validation_0.1.1.json). Esse relatório documenta execução do pacote extraído em processo Python isolado, em autojogo: 720 estados registrados, ambos os jogadores em `DONE`, listas de erros e estados inválidos vazias. Ele verifica execução e isolamento do artefato; um único autojogo não mede força frente a outros agentes.

O motor usado tem SHA-256 `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`, correspondente ao snapshot fixado documentado em [ENGINE_NOTES.md](../tecnica/ENGINE_NOTES.md).

## Comparação intermediária e diagnósticos

[benchmark_0.1.1_comparison.json](../../outputs/benchmark_0.1.1_comparison.json) contém uma comparação histórica de `adaptive` e `baseline_010` contra `starter`, nas seeds 11, 29 e 47, com os dois lados. O próprio relatório identifica o candidato como `source="working_tree"` e `frozen=false`.

**Esse candidato é um snapshot intermediário, anterior ao pacote final preservado.** A conferência dos hashes mostra diferenças em `kaggriculture_agent/agent.py` e `kaggriculture_agent/market.py` em relação ao arquivo final. Portanto, seus saldos, margens, tempos e ratings devem permanecer atribuídos àquele snapshot; não são resultados certificados do `baseline_011` final nem da 0.1.2.

Os arquivos [v011-initial-smoke.json](../../outputs/v011-initial-smoke.json) e [v011-corrected-smoke.json](../../outputs/v011-corrected-smoke.json) também são diagnósticos de desenvolvimento. A identificação de falhas e correções que motivaram alterações está nas [notas de implementação da 0.1.1](../versoes/RELEASE_0.1.1.md).

## Comparações reproduzíveis posteriores

`baseline_011` carrega o pacote preservado em namespace isolado. Para avaliar novamente a referência final no ambiente preparado:

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark --variants baseline_011 --seeds 11 29 47 --opponents starter --output outputs/baseline_011_recheck.json
```

Esse comando produz um novo relatório; sua presença aqui não afirma que essa bateria específica já foi executada. Os confrontos recentes entre a política atual e a referência congelada estão reunidos no [índice de resultados da 0.1.2](../README.md#resultados-da-012). O relatório [RESULTADOS.md](RESULTADOS.md) permanece como histórico da **0.1.0**.
