<#
.SYNOPSIS
  Convert a PEFT LoRA adapter to a llama.cpp/LM Studio-compatible GGUF LoRA (Adapter Mode),
  or MERGE the LoRA into an HF base and then export a merged GGUF model (Merge Mode).

.DESCRIPTION
  - Prefer Adapter Mode using llama.cpp's official converters:
      convert-lora-to-gguf.py   (for LoRA adapters)
      convert-hf-to-gguf.py     (for merged HF checkpoints)
  - If these scripts aren't available, the script will explain how to install/point to them.

.PARAMETER PeftPath
  Path to the PEFT LoRA directory containing adapter_model.safetensors and adapter_config.json.

.PARAMETER OutputDir
  Directory to place outputs. For Adapter Mode: *.gguf (LoRA adapter). For Merge Mode: merged *.gguf model.

.PARAMETER LlamaCppTools
  Path to a local llama.cpp repo (or its 'convert' scripts folder) containing:
    convert-lora-to-gguf.py and convert-hf-to-gguf.py

.PARAMETER BaseModelHF
  (Optional) Path to a local HuggingFace-format base model directory for "gemma-3-270m-it".
  Required for Merge Mode, optional for Adapter Mode (but strongly recommended to ensure correct arch).

.PARAMETER Mode
  'Adapter' (default) or 'Merge'.
    Adapter -> produce a compact LoRA .gguf adapter applied on top of the base GGUF in LM Studio.
    Merge   -> merge LoRA into HF base, then export a full merged GGUF (larger file).

.EXAMPLE
  # Adapter Mode with local llama.cpp tools and local HF base (recommended)
  .\convert_lora_to_gguf.ps1 `
     -PeftPath .\artifacts\ft-gemma3-270m-code-lora `
     -OutputDir .\artifacts `
     -LlamaCppTools C:\dev\llama.cpp `
     -BaseModelHF C:\models\gemma-3-270m-it-hf `
     -Mode Adapter

.EXAMPLE
  # Merge Mode: produces a new merged base .gguf (large, but simple to deploy)
  .\convert_lora_to_gguf.ps1 `
     -PeftPath .\artifacts\ft-gemma3-270m-code-lora `
     -OutputDir .\artifacts `
     -LlamaCppTools C:\dev\llama.cpp `
     -BaseModelHF C:\models\gemma-3-270m-it-hf `
     -Mode Merge

#>

param(
  [Parameter(Mandatory = $true)][string]$PeftPath,
  [Parameter(Mandatory = $true)][string]$OutputDir,
  [Parameter(Mandatory = $true)][string]$LlamaCppTools,
  [string]$BaseModelHF = "",
  [ValidateSet("Adapter","Merge")][string]$Mode = "Adapter"
)

function Require-File([string]$Path){
  if (-not (Test-Path $Path)) { throw "Required path not found: $Path" }
}

# Validate inputs
Require-File $PeftPath
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }

$convLora = Join-Path $LlamaCppTools "convert-lora-to-gguf.py"
$convHF   = Join-Path $LlamaCppTools "convert-hf-to-gguf.py"

if ($Mode -eq "Adapter") {
  if (-not (Test-Path $convLora)) {
    Write-Warning "convert-lora-to-gguf.py not found at: $convLora"
    Write-Host "Clone llama.cpp and set -LlamaCppTools to its path. Example:" -ForegroundColor Yellow
    Write-Host "  git clone https://github.com/ggerganov/llama.cpp C:\dev\llama.cpp" -ForegroundColor Yellow
    exit 1
  }

  $adapterFile = Join-Path $PeftPath "adapter_model.safetensors"
  Require-File $adapterFile
  $outFile = Join-Path $OutputDir "lora-gemma3-270m-it.gguf"

  $cmd = @("python", $convLora, "--i", $adapterFile, "--o", $outFile)
  if ($BaseModelHF -and (Test-Path $BaseModelHF)) {
    $cmd += @("--lora-base", $BaseModelHF)
  } else {
    Write-Warning "No -BaseModelHF provided. The converter will try to infer architecture. If it fails, supply -BaseModelHF."
  }

  Write-Host "Running: $($cmd -join ' ')" -ForegroundColor Cyan
  $p = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Count-1)] -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "Adapter conversion failed with exit code $($p.ExitCode)." }

  Write-Host "Success! GGUF LoRA adapter: $outFile" -ForegroundColor Green
  Write-Host "Use in LM Studio / llama.cpp with the base GGUF and this --lora file." -ForegroundColor Green
  exit 0
}

if ($Mode -eq "Merge") {
  if (-not (Test-Path $convHF)) {
    Write-Warning "convert-hf-to-gguf.py not found at: $convHF"
    Write-Host "Clone llama.cpp and set -LlamaCppTools to its path. Example:" -ForegroundColor Yellow
    Write-Host "  git clone https://github.com/ggerganov/llama.cpp C:\dev\llama.cpp" -ForegroundColor Yellow
    exit 1
  }
  if (-not $BaseModelHF) { throw "Merge mode requires -BaseModelHF (HF directory of gemma-3-270m-it)." }

  $mergePy = Join-Path $PSScriptRoot "merge_lora_and_export_hf.py"
  Require-File $mergePy

  $mergedHF = Join-Path $OutputDir "merged-gemma3-270m-it-hf"
  if (-not (Test-Path $mergedHF)) { New-Item -ItemType Directory -Path $mergedHF | Out-Null }

  # 1) Merge PEFT into base HF
  $mergeCmd = @("python", $mergePy, "--base", $BaseModelHF, "--peft", $PeftPath, "--out", $mergedHF)
  Write-Host "Merging LoRA into base HF:" -ForegroundColor Cyan
  Write-Host "  $($mergeCmd -join ' ')" -ForegroundColor Cyan
  $p1 = Start-Process -FilePath $mergeCmd[0] -ArgumentList $mergeCmd[1..($mergeCmd.Count-1)] -Wait -PassThru
  if ($p1.ExitCode -ne 0) { throw "HF merge failed with exit code $($p1.ExitCode)." }

  # 2) Convert merged HF to GGUF
  $outGGUF = Join-Path $OutputDir "merged-gemma3-270m-it.gguf"
  $convCmd = @("python", $convHF, "--outfile", $outGGUF, $mergedHF)
  Write-Host "Converting merged HF -> GGUF:" -ForegroundColor Cyan
  Write-Host "  $($convCmd -join ' ')" -ForegroundColor Cyan
  $p2 = Start-Process -FilePath $convCmd[0] -ArgumentList $convCmd[1..($convCmd.Count-1)] -Wait -PassThru
  if ($p2.ExitCode -ne 0) { throw "HF->GGUF conversion failed with exit code $($p2.ExitCode)." }

  Write-Host "Success! Merged GGUF model: $outGGUF" -ForegroundColor Green
  exit 0
}
