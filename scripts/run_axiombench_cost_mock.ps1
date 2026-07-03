# Reproduce the AxiomBench cost pillar without requiring an upstream API key.
#
param(
    [int]$ProxyPort = 3100,
    [int]$MockPort = 3101,
    [string]$Out = '..\bench\results\phase4-live-mock.json'
)

# Starts a tiny local /v1/responses mock, starts an isolated Axiom proxy pointed
# at that mock, runs `axiombench --live`, and cleans up both local processes.
$ErrorActionPreference = 'Stop'

$Repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$Crate = Join-Path $Repo 'axiom_engine_rs'
$Engine = Join-Path $Crate 'target-test\debug\axiom_engine.exe'

Push-Location $Crate
try {
    $env:CARGO_INCREMENTAL = '0'
    $env:CARGO_TARGET_DIR = 'target-test'
    cargo build --features tools --bin axiom_engine --bin axiombench --locked
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$mockJob = Start-Job -ScriptBlock {
    param($Port)
    $listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Parse('127.0.0.1'),
        $Port
    )
    $listener.Start()
    try {
        while ($true) {
            $client = $listener.AcceptTcpClient()
            try {
                $stream = $client.GetStream()
                $buffer = New-Object byte[] 65536
                $null = $stream.Read($buffer, 0, $buffer.Length)
                $body = '{"id":"resp_mock","object":"response","output":[],"status":"completed"}'
                $response = "HTTP/1.1 200 OK`r`nContent-Type: application/json`r`nContent-Length: $($body.Length)`r`nConnection: close`r`n`r`n$body"
                $bytes = [System.Text.Encoding]::ASCII.GetBytes($response)
                $stream.Write($bytes, 0, $bytes.Length)
            } finally {
                $client.Close()
            }
        }
    } finally {
        $listener.Stop()
    }
} -ArgumentList $MockPort

$proxy = $null
try {
    $mockReady = $false
    for ($i = 1; $i -le 50; $i++) {
        if ($mockJob.State -ne 'Running') {
            $mockError = Receive-Job $mockJob -ErrorAction SilentlyContinue | Out-String
            throw "Responses mock did not start on port $MockPort. $mockError"
        }
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $client.BeginConnect('127.0.0.1', $MockPort, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(200)) {
                $client.EndConnect($connect)
                $mockReady = $true
                break
            }
        } catch {
        } finally {
            $client.Close()
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $mockReady) {
        throw "Responses mock did not become ready on port $MockPort"
    }

    $env:AXIOM_PRODUCTION_BPE = '1'
    $env:AXIOM_TOKENIZER = Join-Path $Repo 'checkpoints\axiom_bpe.json'
    $env:AXIOM_BPE_CKPT = Join-Path $Repo 'checkpoints\axiom_production_bpe.bin'
    $env:AXIOM_MEMORY_DIR = Join-Path $Repo 'checkpoints\memory'
    $env:AXIOM_TTT_COMPRESS = '1'
    $env:AXIOM_TTT_COMPRESS_THRESHOLD_TOKENS = '128'
    $env:AXIOM_OPENAI_RESPONSES_UPSTREAM = "http://127.0.0.1:$MockPort/v1/responses"

    $proxyArgs = @(
        '--mode', 'server',
        '--host', '127.0.0.1',
        '--port', "$ProxyPort",
        '--checkpoint', (Join-Path $Repo 'checkpoints\axiom_production_bpe.bin'),
        '--device', 'cpu'
    )
    $proxy = Start-Process -FilePath $Engine `
        -ArgumentList $proxyArgs `
        -WorkingDirectory $Repo `
        -PassThru `
        -WindowStyle Hidden

    $ready = $false
    for ($i = 1; $i -le 90; $i++) {
        try {
            $response = Invoke-WebRequest `
                -UseBasicParsing `
                -Uri "http://127.0.0.1:$ProxyPort/healthz" `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        throw "Axiom proxy did not become healthy on port $ProxyPort"
    }

    Push-Location $Crate
    try {
        $env:AXIOM_BASE_URL = "http://127.0.0.1:$ProxyPort"
        cargo run --features tools --bin axiombench --locked -- --live --out $Out
        if ($LASTEXITCODE -ne 0) {
            throw "axiombench live replay failed with exit code $LASTEXITCODE"
        }

        $resultPath = Resolve-Path $Out
        $result = Get-Content $resultPath -Raw | ConvertFrom-Json
        $cost = $result.results | Where-Object { $_.name -eq 'cost' } | Select-Object -First 1
        if (-not $cost -or [int]$cost.detail.replayed -le 0) {
            throw "Live cost replay produced no successful replays; see $resultPath"
        }
    } finally {
        Pop-Location
    }
} finally {
    if ($proxy -and -not $proxy.HasExited) {
        Stop-Process -Id $proxy.Id -Force
    }
    Stop-Job $mockJob -ErrorAction SilentlyContinue
    Remove-Job $mockJob -Force -ErrorAction SilentlyContinue
}
