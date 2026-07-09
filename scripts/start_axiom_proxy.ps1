# start_axiom_proxy.ps1 — Windows-native boot of the Axiom-TTT context-compression proxy.
#
# This is the PowerShell equivalent of start_axiom.sh, designed for Windows machines
# that may not have Git Bash reliably available. It launches the Rust axiom_engine
# server in --mode server with the full compression/forwarding environment.
#
# The server exposes Anthropic-compatible POST /v1/messages and OpenAI-compatible
# POST /v1/chat/completions endpoints. When compression is enabled it absorbs
# "heavy" context locally (TTT) and forwards a lean, fingerprinted payload to the
# REAL upstream API.
#
# IMPORTANT — upstream vs. client routing
# ----------------------------------------
# The forwarder reads ANTHROPIC_BASE_URL to choose ITS OWN upstream. If you point
# ANTHROPIC_BASE_URL at this proxy (127.0.0.1:3000) in the SAME shell that runs
# the server, the proxy forwards to itself -> infinite loop. This script pins the
# server's upstream to the real Anthropic API and ignores any inherited client value.
# Client redirection belongs in a SEPARATE shell — see axiom.env.
#
# Usage:
#   .\scripts\start_axiom_proxy.ps1
#   .\scripts\start_axiom_proxy.ps1 -Port 3000 -Host 127.0.0.1
#   .\scripts\start_axiom_proxy.ps1 -Watchdog  (restart on crash)
#
param(
    [string]$BindHost = $(if ($env:AXIOM_HOST) { $env:AXIOM_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:AXIOM_PORT) { [int]$env:AXIOM_PORT } else { 3000 }),
    [switch]$Watchdog = $false,
    [switch]$NoWatchdog = $false
)

$ErrorActionPreference = 'Stop'

# --- Resolve paths ---------------------------------------------------------
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Bin = Join-Path $RepoRoot "axiom_engine_rs\target\release\axiom_engine.exe"
$Checkpoint = Join-Path $RepoRoot "checkpoints\axiom_production_bpe.bin"
$Tokenizer = Join-Path $RepoRoot "checkpoints\axiom_bpe.json"
$LogOut = Join-Path $RepoRoot "axiom_server.log"
$LogErr = Join-Path $RepoRoot "axiom_server.err.log"

# --- Verify binary exists --------------------------------------------------
if (-not (Test-Path $Bin)) {
    Write-Host "[start_axiom_proxy] FATAL: Release binary missing: $Bin"
    Write-Host "[start_axiom_proxy] Build it first:  cargo build --release --manifest-path `"$RepoRoot\axiom_engine_rs\Cargo.toml`""
    exit 1
}

# --- Network boundary ------------------------------------------------------
$Upstream = $(if ($env:AXIOM_UPSTREAM_URL) { $env:AXIOM_UPSTREAM_URL } else { "https://api.anthropic.com" })
if ($Upstream -match "127\.0\.0\.1:$Port" -or $Upstream -match "localhost:$Port") {
    Write-Host "[start_axiom_proxy] FATAL: upstream ($Upstream) points back at this proxy ($BindHost:$Port)."
    Write-Host "[start_axiom_proxy] That would create an infinite forward loop. Set AXIOM_UPSTREAM_URL"
    Write-Host "[start_axiom_proxy] to the real Anthropic API (https://api.anthropic.com)."
    exit 1
}

# --- Save current env so we can restore after launch -----------------------
$oldEnv = @{
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
    # --- Compression config ----------------------------------------------------
    $env:AXIOM_TTT_COMPRESS = $(if ($env:AXIOM_TTT_COMPRESS) { $env:AXIOM_TTT_COMPRESS } else { "1" })
    $env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS = $(if ($env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS) { $env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS } else { "200" })
    $env:AXIOM_TTT_COMPRESS_TOP_K = $(if ($env:AXIOM_TTT_COMPRESS_TOP_K) { $env:AXIOM_TTT_COMPRESS_TOP_K } else { "32" })
    $env:AXIOM_RESPONSES_COMPRESS = "1"

    # --- Compute device: CPU on this machine (no CUDA toolkit) -----------------
    $env:AXIOM_DEVICE = $(if ($env:AXIOM_DEVICE) { $env:AXIOM_DEVICE } else { "cpu" })

    # --- Production BPE model --------------------------------------------------
    if ((Test-Path $Checkpoint) -and (Test-Path $Tokenizer)) {
        $env:AXIOM_PRODUCTION_BPE = "1"
        $env:AXIOM_TOKENIZER = $Tokenizer
        $env:AXIOM_BPE_CKPT = $Checkpoint

        # Use eval-calibrated drift gate if available, otherwise default
        $GateFile = Join-Path $RepoRoot "checkpoints\axiom_drift_gate.txt"
        if (Test-Path $GateFile) {
            $env:AXIOM_DRIFT_THRESHOLD = (Get-Content $GateFile -Raw).Trim()
        } else {
            $env:AXIOM_DRIFT_THRESHOLD = $(if ($env:AXIOM_DRIFT_THRESHOLD) { $env:AXIOM_DRIFT_THRESHOLD } else { "7.03" })
        }
        Write-Host "[start_axiom_proxy] Production model: BPE semantic (drift_gate=$($env:AXIOM_DRIFT_THRESHOLD))"
    } else {
        Write-Host "[start_axiom_proxy] Production model: legacy 256-hash (BPE artifacts not found)"
    }

    # --- Vibe memory -----------------------------------------------------------
    # Enable persistent fast-weight memory by default in this script.
    # Set AXIOM_VIBE=0 before running to disable.
    $env:AXIOM_VIBE = $(if ($null -ne $env:AXIOM_VIBE) { $env:AXIOM_VIBE } else { "1" })

    # --- Memory store ----------------------------------------------------------
    $env:AXIOM_MEMORY_DIR = Join-Path $RepoRoot "checkpoints\memory"

    # --- Server upstream must be the real provider, never this proxy -----------
    $env:ANTHROPIC_BASE_URL = $Upstream

    # --- Checkpoint args -------------------------------------------------------
    $CkptArgs = @()
    if (Test-Path $Checkpoint) {
        Write-Host "[start_axiom_proxy] Using production checkpoint: $Checkpoint"
        $CkptArgs = @("--checkpoint", $Checkpoint)
    } else {
        Write-Host "[start_axiom_proxy] WARNING: $Checkpoint not found — booting with fresh init"
        Write-Host "[start_axiom_proxy]          (small CPU model, low-fidelity compression fingerprint)"
    }

    # --- Preflight: check for ANTHROPIC_API_KEY -------------------------------
    if ($env:AXIOM_TTT_COMPRESS -eq "1" -and -not $env:ANTHROPIC_API_KEY) {
        Write-Host "[start_axiom_proxy] No ANTHROPIC_API_KEY set -> AUTH-PASSTHROUGH mode."
        Write-Host "[start_axiom_proxy]   The proxy holds no key of its own and relays each client's"
        Write-Host "[start_axiom_proxy]   own Authorization / x-api-key headers upstream."
    }

    Write-Host "[start_axiom_proxy] Launching Axiom-TTT proxy"
    Write-Host "[start_axiom_proxy]   bind        : http://$BindHost`:$Port"
    Write-Host "[start_axiom_proxy]   upstream    : $env:ANTHROPIC_BASE_URL"
    Write-Host "[start_axiom_proxy]   compression : $env:AXIOM_TTT_COMPRESS (threshold=$($env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS) tokens, top_k=$($env:AXIOM_TTT_COMPRESS_TOP_K))"
    Write-Host "[start_axiom_proxy]   device      : $env:AXIOM_DEVICE"
    Write-Host "[start_axiom_proxy]   vibe        : $env:AXIOM_VIBE"
    Write-Host "[start_axiom_proxy]   log         : $LogOut"
    Write-Host ""

    $AllArgs = @("--mode", "server", "--host", $BindHost, "--port", $Port) + $CkptArgs

    if ($Watchdog -and -not $NoWatchdog) {
        # --- Watchdog: restart on crash ----------------------------------------
        $Attempt = 0
        while ($true) {
            $Attempt++
            Write-Host "[watchdog] Starting axiom_engine (attempt #$Attempt, device=$($env:AXIOM_DEVICE))"

            $proc = Start-Process -FilePath $Bin `
                -ArgumentList $AllArgs `
                -WorkingDirectory $RepoRoot `
                -WindowStyle Hidden `
                -RedirectStandardOutput $LogOut `
                -RedirectStandardError $LogErr `
                -PassThru

            $proc.WaitForExit()
            $ExitCode = $proc.ExitCode
            Write-Host "[watchdog] axiom_engine exited (code=$ExitCode) at $(Get-Date)"

            # Check for CUDA errors in log
            $logContent = ""
            if (Test-Path $LogErr) { $logContent = Get-Content $LogErr -Raw -ErrorAction SilentlyContinue }
            if ($logContent -match "CUDA_ERROR|cuda error|DriverError") {
                Write-Host "[watchdog] CUDA error detected in log — switching to CPU for next run"
                $env:AXIOM_DEVICE = "cpu"
            }

            Write-Host "[watchdog] Restarting in 3 seconds..."
            Start-Sleep -Seconds 3
        }
    } else {
        # --- Single launch (no watchdog) ---------------------------------------
        $proc = Start-Process -FilePath $Bin `
            -ArgumentList $AllArgs `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $LogOut `
            -RedirectStandardError $LogErr `
            -PassThru

        Write-Host "[start_axiom_proxy] axiom_engine started (PID $($proc.Id))"

        # Wait for port to be ready
        Write-Host "[start_axiom_proxy] Waiting for port $Port..."
        for ($i = 1; $i -le 30; $i++) {
            Start-Sleep -Seconds 1
            $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($conn) {
                Write-Host "[start_axiom_proxy]   PROXY UP after ${i}s (PID $($conn.OwningProcess))"
                break
            }
        }

        # Verify config
        try {
            $config = Invoke-RestMethod -Uri "http://${BindHost}:$Port/v1/config" -TimeoutSec 5
            Write-Host "[start_axiom_proxy]   compression_active=$($config.compression_active) forwarder_ready=$($config.forwarder_ready)"
        } catch {
            Write-Host "[start_axiom_proxy]   WARNING: could not read /v1/config — proxy may still be starting"
        }
    }

} finally {
    # Restore original env vars
    foreach ($name in $oldEnv.Keys) {
        if ($null -eq $oldEnv[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item "Env:$name" $oldEnv[$name]
        }
    }
}