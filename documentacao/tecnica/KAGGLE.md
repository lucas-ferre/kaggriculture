# Kaggle CLI e preparação da submissão

Execute os comandos no PowerShell, a partir da raiz do projeto. O CLI oficial está instalado no ambiente virtual `.venv`, com versão fixada em [requirements-kaggle.txt](../../requirements-kaggle.txt).

## Instalação e credenciais

Para reinstalar o CLI no ambiente preparado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-kaggle.txt
```

O arquivo `kaggle.json` deve permanecer na raiz do projeto, com os campos `username` e `key` fornecidos pelo Kaggle. O [atalho scripts/kaggle.ps1](../../scripts/kaggle.ps1) define `KAGGLE_CONFIG_DIR` para essa pasta durante a execução e chama o CLI do ambiente virtual. Ele restaura as variáveis da sessão ao terminar. A credencial já está coberta pelo `.gitignore` e fica fora do pacote de submissão.

O mecanismo de autenticação está descrito na [documentação oficial do Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication). Credenciais podem ser geradas nas [configurações de API da conta](https://www.kaggle.com/settings/api).

## Testar acesso sem enviar arquivos

```powershell
.\scripts\kaggle.ps1 competitions list --group entered -s kaggriculture
.\scripts\kaggle.ps1 competitions submissions kaggriculture
```

As duas operações são consultas. A primeira permite conferir a inscrição; a segunda consulta o histórico de submissões. Um histórico vazio pode ser uma resposta válida.

Na verificação mais recente, registrada em [kaggle_auth_check.json](../../outputs/kaggle_auth_check.json), a credencial renovada foi aceita e as duas consultas foram concluídas com sucesso. A conta está inscrita em `kaggriculture`, e o histórico consultado não contém submissões. Nenhuma submissão foi realizada nesta configuração.

## Testar o pacote localmente

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m scripts.package_submission
.\.venv\Scripts\python.exe -m scripts.validate_submission --output outputs/pre_submission_validation_0.1.2.json
```

Na preparação atual, os 160 testes passaram. O pacote extraído foi executado em processo Python isolado por 720 estados, com os dois jogadores em `DONE` e sem erros. O [relatório de validação](../../outputs/pre_submission_validation_0.1.2.json) identifica o SHA-256 do arquivo testado. Essa execução é local; a validação hospedada acontece após uma submissão efetiva.

## Comando para envio manual

Com a autenticação e a inscrição confirmadas, este comando envia o pacote:

```powershell
.\scripts\kaggle.ps1 competitions submit kaggriculture -f .\dist\submission.tar.gz -m "Kaggriculture agent 0.1.2"
```

Alternativa sem o atalho PowerShell:

```powershell
$env:KAGGLE_CONFIG_DIR = (Get-Location).Path
.\.venv\Scripts\kaggle.exe competitions submit kaggriculture -f .\dist\submission.tar.gz -m "Kaggriculture agent 0.1.2"
```

Os comandos de envio acima foram apenas documentados. A sintaxe segue os [comandos oficiais de competições](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md). Envie apenas o arquivo gerado em `dist/`; ele contém o agente e as licenças, sem credenciais, documentação ou ambiente virtual.
