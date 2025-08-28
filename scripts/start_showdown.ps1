# scripts\start_showdown.ps1
$ErrorActionPreference = "Stop"

# cd to repo
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..\showdown

# Ensure config exists (correct filename in repo is config-example.js)
$configDir  = Join-Path (Get-Location) "config"
$configSrc  = Join-Path $configDir "config-example.js"
$configPath = Join-Path $configDir "config.js"

if (-not (Test-Path $configPath)) {
  if (-not (Test-Path $configSrc)) {
    throw "Could not find $configSrc — is the Showdown repo fully cloned? Try: git clone https://github.com/smogon/pokemon-showdown.git showdown && npm install"
  }
  Copy-Item $configSrc $configPath
}

# Move logs to %TEMP% (avoid OneDrive churn) + cut verbosity
$logRoot = Join-Path $env:TEMP "ps-logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

# Idempotently enforce lightweight logging
$override = @"
exports.loglevel = 'error';
exports.battleLogFormat = 'none';
exports.logfilepath = '$logRoot';
"@

# Append only if not present
$content = Get-Content $configPath -Raw
if ($content -notmatch "battleLogFormat") {
  Add-Content -Path $configPath -Value $override
}

# Launch
node pokemon-showdown start --no-security --no-log-color
