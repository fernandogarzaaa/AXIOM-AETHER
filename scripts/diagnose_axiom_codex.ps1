param(
    [string]$ProxyUrl = "http://127.0.0.1:3000",
    [string]$SessionId = "mcp",
    [string]$AxiomExe = ".\axiom_engine_rs\target\release\axiom_engine.exe",
    [string]$Checkpoint = ".\checkpoints\axiom_production_bpe.bin"
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Name)
    Write-Host ""
    Write-Host "== $Name =="
}

function Invoke-JsonRpc {
    param(
        [string]$Method,
        [object]$Params = @{}
    )
    $payload = @{
        jsonrpc = "2.0"
        id      = 1
        method  = $Method
        params  = $Params
    } | ConvertTo-Json -Depth 8 -Compress
    $payload | & $AxiomExe --mode mcp --checkpoint $Checkpoint 2>$null
}

Write-Section "HTTP proxy"
foreach ($path in @("/healthz", "/readyz", "/metrics", "/v1/config")) {
    $url = "$($ProxyUrl.TrimEnd('/'))$path"
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5
        Write-Host "$path HTTP $($resp.StatusCode)"
        if ($path -eq "/v1/config") {
            Write-Host $resp.Content
        }
    } catch {
        Write-Host "$path ERROR $($_.Exception.Message)"
    }
}

Write-Section "Awareness"
try {
    $url = "$($ProxyUrl.TrimEnd('/'))/v1/awareness/$SessionId"
    $resp = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5
    Write-Host $resp.Content
} catch {
    Write-Host "awareness ERROR $($_.Exception.Message)"
    Write-Host "If this is a fresh Codex MCP session, call axiom_status or POST /v1/budget to initialize awareness."
}

Write-Section "MCP tools"
if (!(Test-Path -LiteralPath $AxiomExe)) {
    Write-Host "MCP binary not found: $AxiomExe"
} elseif (!(Test-Path -LiteralPath $Checkpoint)) {
    Write-Host "Checkpoint not found: $Checkpoint"
} else {
    try {
        $tools = Invoke-JsonRpc -Method "tools/list" | ConvertFrom-Json
        $names = @($tools.result.tools | ForEach-Object { $_.name })
        Write-Host "tool_count=$($names.Count)"
        Write-Host ($names -join ", ")
        foreach ($required in @("axiom_compress_path", "axiom_expand", "axiom_remember", "axiom_recall", "axiom_verify", "axiom_evaluate_drift", "axiom_status", "search", "fetch")) {
            if ($names -notcontains $required) {
                Write-Host "MISSING_TOOL $required"
            }
        }
    } catch {
        Write-Host "MCP tools/list ERROR $($_.Exception.Message)"
    }
}

Write-Section "Verifier smoke"
try {
    $params = @{
        name      = "axiom_verify"
        arguments = @{
            response = "pytest passed with 2 tests."
            evidence = "pytest output was '.. [100%] 2 passed in 1.66s'."
        }
    }
    Invoke-JsonRpc -Method "tools/call" -Params $params
} catch {
    Write-Host "verifier smoke ERROR $($_.Exception.Message)"
}

Write-Section "Notes"
Write-Host "Compression fingerprints with compression_confidence=low should be treated as advisory only."
Write-Host "Drift reports with drift=UNAVAILABLE mean no master-vibe baseline exists yet; run axiom_compress_path on source first."
