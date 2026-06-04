# restart_proxy.ps1 — definitively stop every Axiom proxy/watchdog instance and
# start exactly one, pinned to CPU for stability. Idempotent and quote-safe.
$ErrorActionPreference = 'SilentlyContinue'

Write-Host '== stopping all proxy + watchdog instances =='
Get-CimInstance Win32_Process -Filter "Name='bash.exe'" |
    Where-Object { $_.CommandLine -match 'start_axiom' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Get-Process axiom_engine | Stop-Process -Force
Start-Sleep -Seconds 2

$alive = (Get-Process axiom_engine | Measure-Object).Count
$wd = (Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -match 'start_axiom' } | Measure-Object).Count
Write-Host "  axiom_engine=$alive  watchdog=$wd"

Write-Host '== launching one clean proxy (CPU) =='
$bash = 'C:\Program Files\Git\bin\bash.exe'
Start-Process -FilePath $bash `
    -ArgumentList '-lc', 'cd /c/Users/garza/AXIOM-AETHER && AXIOM_DEVICE=cpu AXIOM_VIBE=0 ./start_axiom.sh' `
    -WindowStyle Hidden

Write-Host '== waiting for port 3000 =='
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    $c = Get-NetTCPConnection -LocalPort 3000 -State Listen
    if ($c) { Write-Host "  PROXY UP after ${i}s (PID $($c.OwningProcess | Select-Object -First 1))"; break }
}
