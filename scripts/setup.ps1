# scripts\setup.ps1
# PowerShell-native setup for metagross (Windows-only)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PyExe {
  # Prefer the py launcher with 3.11; else try python
  $candidates = @(
    { & py -3.11 -c "import sys; print(sys.executable)" },
    { & py -3.12 -c "import sys; print(sys.executable)" },
    { & python -c "import sys; print(sys.executable)" }
  )
  foreach ($try in $candidates) {
    try {
      $out = & $try 2>$null
      if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
    } catch {}
  }
  throw "Could not find Python. Install Python 3.11+ or the 'py' launcher."
}

# Move to project root (metagross)
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location ..

# 1) Ensure local venv
if (-not (Test-Path ".\.venv")) {
  $py = Get-PyExe
  & $py -m venv .venv
}

$pip = ".\.venv\Scripts\pip.exe"
$python = ".\.venv\Scripts\python.exe"

# 2) Upgrade pip
& $python -m pip install --upgrade pip

# 3) Install PyTorch (CUDA 12.4) separately (don’t rely on requirements.txt for this)
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4) Install the rest (lightweight requirements)
if (-not (Test-Path ".\requirements.txt")) {
@'
poke-env==0.5.2
numpy
pandas
tqdm
'@ | Out-File -Encoding utf8 .\requirements.txt
}
& $pip install -r .\requirements.txt

# 5) Clone Pokémon Showdown if missing
if (-not (Test-Path ".\showdown")) {
  git clone https://github.com/smogon/pokemon-showdown.git showdown
}

# 6) Install Node deps
Set-Location .\showdown
if (Test-Path "package-lock.json") {
  npm ci
} else {
  npm install
}

Write-Host "Setup complete." -ForegroundColor Green
