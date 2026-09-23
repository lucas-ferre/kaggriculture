# Run the project-local Kaggle CLI with the credentials from the project root.
# Example (read-only): .\scripts\kaggle.ps1 competitions submissions kaggriculture
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$kaggleExecutable = Join-Path $projectRoot '.venv/Scripts/kaggle.exe'
$credentialFile = Join-Path $projectRoot 'kaggle.json'

if (-not (Test-Path -LiteralPath $kaggleExecutable -PathType Leaf)) {
    throw 'Install the Kaggle CLI with .\.venv\Scripts\python.exe -m pip install -r requirements-kaggle.txt.'
}
if (-not (Test-Path -LiteralPath $credentialFile -PathType Leaf)) {
    throw 'Place your kaggle.json file in the project root.'
}

$previousConfigDirectory = $env:KAGGLE_CONFIG_DIR
$previousPythonUtf8 = $env:PYTHONUTF8
$kaggleArguments = $args
if ($kaggleArguments.Count -eq 0) {
    $kaggleArguments = @('--help')
}
try {
    $env:KAGGLE_CONFIG_DIR = $projectRoot
    $env:PYTHONUTF8 = '1'
    & $kaggleExecutable @kaggleArguments
    $kaggleExitCode = $LASTEXITCODE
}
finally {
    $env:KAGGLE_CONFIG_DIR = $previousConfigDirectory
    $env:PYTHONUTF8 = $previousPythonUtf8
}
exit $kaggleExitCode
