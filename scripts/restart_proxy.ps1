# restart_proxy.ps1 - restart only the main Axiom HTTP proxy on port 3000.
#
# Claude/Codex can also spawn axiom_engine.exe in stdio MCP mode, and the
# ChatGPT connector runs its own HTTP MCP server on port 8083. Do not kill those
# processes here; stopping all axiom_engine.exe instances interrupts active
# agent sessions. This script only replaces the process that owns port 3000 and
# launches it with the full compression/forwarding environment.
$ErrorActionPreference = 'Stop'

$Root = 'D:\AXIOM-AETHER'
$Bin = Join-Path $Root 'axiom_engine_rs\target\release\axiom_engine.exe'
$Checkpoint = Join-Path $Root 'checkpoints\axiom_production_bpe.bin'
$Tokenizer = Join-Path $Root 'checkpoints\axiom_bpe.json'
$LogOut = Join-Path $Root 'axiom_server.log'
$LogErr = Join-Path $Root 'axiom_server.err.log'

Write-Host '== stopping main proxy on port 3000 only =='
Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
    Where-Object { $_.CommandLine -match 'start_axiom' -and $_.CommandLine -match '3000|AXIOM_PORT' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
        $proc = Get-Process -Id $_ -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -eq 'axiom_engine') {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $proc.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
    }
Start-Sleep -Seconds 1

Write-Host '== launching main proxy with compression enabled =='
$old = @{
    ANTHROPIC_BASE_URL = $env:ANTHROPIC_BASE_URL
    AXIOM_TTT_COMPRESS = $env:AXIOM_TTT_COMPRESS
    AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS = $env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS
    AXIOM_TTT_COMPRESS_TOP_K = $env:AXIOM_TTT_COMPRESS_TOP_K
    AXIOM_RESPONSES_COMPRESS = $env:AXIOM_RESPONSES_COMPRESS
    AXIOM_DEVICE = $env:AXIOM_DEVICE
    AXIOM_PRODUCTION_BPE = $env:AXIOM_PRODUCTION_BPE
    AXIOM_TOKENIZER = $env:AXIOM_TOKENIZER
    AXIOM_BPE_CKPT = $env:AXIOM_BPE_CKPT
    AXIOM_MEMORY_DIR = $env:AXIOM_MEMORY_DIR
    AXIOM_VIBE = $env:AXIOM_VIBE
    AXIOM_DRIFT_THRESHOLD = $env:AXIOM_DRIFT_THRESHOLD
}

try {
    # Server-side upstream must be the real provider, never this proxy.
    $env:ANTHROPIC_BASE_URL = 'https://api.anthropic.com'
    $env:AXIOM_TTT_COMPRESS = '1'
    $env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS = '400'
    $env:AXIOM_TTT_COMPRESS_TOP_K = '32'
    $env:AXIOM_RESPONSES_COMPRESS = '1'
    $env:AXIOM_DEVICE = 'cpu'
    $env:AXIOM_PRODUCTION_BPE = '1'
    $env:AXIOM_TOKENIZER = $Tokenizer
    $env:AXIOM_BPE_CKPT = $Checkpoint
    $env:AXIOM_MEMORY_DIR = Join-Path $Root 'checkpoints\memory'
    $env:AXIOM_VIBE = '1'
    $env:AXIOM_DRIFT_THRESHOLD = '7.03'

    Start-Process -FilePath $Bin `
        -ArgumentList @('--mode', 'server', '--host', '127.0.0.1', '--port', '3000', '--checkpoint', $Checkpoint) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogOut `
        -RedirectStandardError $LogErr
} finally {
    foreach ($name in $old.Keys) {
        if ($null -eq $old[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $old[$name]
        }
    }
}

Write-Host '== waiting for port 3000 =='
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    $c = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) {
        Write-Host "  PROXY UP after ${i}s (PID $($c.OwningProcess))"
        break
    }
}

$config = Invoke-RestMethod -Uri 'http://127.0.0.1:3000/v1/config' -TimeoutSec 5
Write-Host "  compression_active=$($config.compression_active) forwarder_ready=$($config.forwarder_ready)"
