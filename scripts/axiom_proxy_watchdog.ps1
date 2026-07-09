# axiom_proxy_watchdog.ps1
# One-shot health check for the main Axiom proxy (port 3000, used by Codex's
# openai_base_url and Claude Code's ANTHROPIC_BASE_URL). Run on a recurring
# Task Scheduler trigger rather than as a long-lived loop process, so the
# watchdog itself can't silently die the way the proxy has twice now.
#
# Launches axiom_engine.exe directly (no bash wrapper) â€” Start-Process on the
# native binary is the one launch method that has proven to survive as an
# independent Windows process on this machine; bash/start_axiom.sh-based
# launches (via nohup+disown or via Start-Process invoking bash) have not.

$Root = "D:\AXIOM-AETHER"
$Bin = "$Root\axiom_engine_rs\target\release\axiom_engine.exe"
$Checkpoint = "$Root\checkpoints\axiom_production_bpe.bin"
$Tokenizer = "$Root\checkpoints\axiom_bpe.json"
$LogOut = "$Root\axiom_server.log"
$LogErr = "$Root\axiom_server.err.log"
$WatchdogLog = "$Root\axiom_proxy_watchdog.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $WatchdogLog -Value "[$ts] $msg"
}

$healthy = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:3000/metrics" -UseBasicParsing -TimeoutSec 5
    if ($r.StatusCode -eq 200) { $healthy = $true }
} catch {
    $healthy = $false
}

if ($healthy) {
    exit 0
}

Write-Log "proxy unhealthy or down - restarting"

# Clear out any stale/half-dead process still claiming port 3000 before relaunch.
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | ForEach-Object {
    $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and $p.ProcessName -eq 'axiom_engine') {
        Write-Log "stopping stale process PID $($p.Id)"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

$env:ANTHROPIC_BASE_URL = "https://api.anthropic.com"
$env:AXIOM_TTT_COMPRESS = "1"
$env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS = "400"
$env:AXIOM_TTT_COMPRESS_TOP_K = "32"
$env:AXIOM_DEVICE = "cpu"
$env:AXIOM_PRODUCTION_BPE = "1"
$env:AXIOM_TOKENIZER = $Tokenizer
$env:AXIOM_BPE_CKPT = $Checkpoint
$env:AXIOM_RESPONSES_COMPRESS = "1"

Start-Process -FilePath $Bin `
    -ArgumentList @("--mode","server","--host","127.0.0.1","--port","3000","--checkpoint",$Checkpoint) `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogOut `
    -RedirectStandardError $LogErr

Start-Sleep -Seconds 5
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:3000/metrics" -UseBasicParsing -TimeoutSec 5
    Write-Log "restart result: HTTP $($r.StatusCode)"
} catch {
    Write-Log "restart result: still unreachable ($($_.Exception.Message))"
}

