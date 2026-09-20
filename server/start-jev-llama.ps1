#requires -Version 7
<#!
.SYNOPSIS
  Start a llama.cpp server tuned for Jev single-decode choice decisions.

.DESCRIPTION
  One request performs a normal prompt prefill and exactly one answer-token
  decision. Speculative decoding is deliberately disabled because there is no
  useful multi-token generation to accelerate.

  The model must support a chat template that can emit A/B/C as the first
  response token. Use -DryRun to inspect the exact command without starting it.
#>
[CmdletBinding()]
param(
    [string]$Model,
    [string]$Alias = 'jev-single-decode',
    [int]$Port = 8080,
    [int]$Context = 8192,
    [int]$Parallel = 1,
    [int]$Batch = 2048,
    [int]$UBatch = 512,
    [int]$Threads = 16,
    [int]$ThreadsBatch = 16,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$exe = (Get-Command llama-server.exe -ErrorAction SilentlyContinue).Source
if (-not $exe) {
    $exe = Join-Path $env:LOCALAPPDATA 'Programs\llama.cpp\llama-server.exe'
}
if (-not (Test-Path $exe)) {
    throw "llama-server.exe was not found. Set PATH or install llama.cpp."
}
if (-not $Model) {
    throw 'Specify -Model C:\path\to\model.gguf.'
}
if (-not (Test-Path $Model)) {
    throw "Model was not found: $Model"
}

$totalContext = $Context * $Parallel
$args = @(
    '--model', $Model
    '--alias', $Alias
    '--host', '127.0.0.1'
    '--port', $Port
    '--ctx-size', $totalContext
    '--parallel', $Parallel
    '--gpu-layers', '999'
    '--flash-attn', 'on'
    '--jinja'
    '--reasoning', 'off'
    '--spec-type', 'none'
    '--batch-size', $Batch
    '--ubatch-size', $UBatch
    '--threads', $Threads
    '--threads-batch', $ThreadsBatch
    '--cache-type-k', 'q8_0'
    '--cache-type-v', 'q8_0'
    '--temperature', '0'
    '--top-k', '0'
    '--top-p', '1'
    '--min-p', '0'
    '--predict', '1'
    '--metrics'
    '--slots'
    '--no-ui'
)

Write-Host "llama.cpp single-decode preset"
Write-Host "  model:   $Model"
Write-Host "  endpoint: http://127.0.0.1:$Port"
Write-Host "  context: $Context per slot x $Parallel = $totalContext"
Write-Host "  batch:   $Batch / ubatch $UBatch"
Write-Host "  threads: $Threads / batch $ThreadsBatch"
Write-Host "  speculative decoding: disabled"
Write-Host "  reasoning: disabled"
Write-Host "  command: $exe $($args -join ' ')"

if ($DryRun) { exit 0 }
& $exe @args
exit $LASTEXITCODE
