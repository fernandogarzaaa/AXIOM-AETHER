param([switch]$Force)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Repo = 'fernandogarzaaa/AXIOM-AETHER'
$VersionFile = Join-Path $Root '.axiom_release_version'
$release = gh release view --repo $Repo --json tagName,assets | ConvertFrom-Json
$tag = $release.tagName
$current = if (Test-Path $VersionFile) { (Get-Content -Raw $VersionFile).Trim() } else { '' }

# A tested post-release build (for example v0.1.6+pr80) is newer than its base
# release and must not be downgraded. A later release still replaces it.
if (!$Force -and ($current -eq $tag -or $current.StartsWith("$tag+"))) {
    & (Join-Path $PSScriptRoot 'restart_proxy.ps1')
    exit $LASTEXITCODE
}

$zipName = "axiom-ttt-$tag-windows-x86_64.zip"
$required = @($zipName, 'axiom_bpe.json', 'axiom_production_bpe.bin', 'axiom_production_bpe.meta.json', 'SHA256SUMS.txt')
$tmp = Join-Path $env:TEMP "axiom-update-$tag"
if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
foreach ($asset in $required) {
    gh release download $tag --repo $Repo --dir $tmp --clobber --pattern $asset
}

$zipAsset = $release.assets | Where-Object name -eq $zipName
$zipHash = (Get-FileHash (Join-Path $tmp $zipName) -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedZip = ($zipAsset.digest -replace '^sha256:', '').ToLowerInvariant()
if (!$expectedZip -or $zipHash -ne $expectedZip) { throw 'Windows release archive SHA-256 mismatch' }

$sums = Get-Content (Join-Path $tmp 'SHA256SUMS.txt')
foreach ($name in @('axiom_bpe.json', 'axiom_production_bpe.bin', 'axiom_production_bpe.meta.json')) {
    $line = $sums | Where-Object { $_ -match "[\\/]$([regex]::Escape($name))$" } | Select-Object -First 1
    if (!$line) { throw "No published SHA-256 for $name" }
    $expected = ($line -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash (Join-Path $tmp $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "$name SHA-256 mismatch" }
}

$expanded = Join-Path $tmp 'expanded'
Expand-Archive -LiteralPath (Join-Path $tmp $zipName) -DestinationPath $expanded -Force
$newExe = Get-ChildItem $expanded -Recurse -Filter axiom_engine.exe | Select-Object -First 1
if (!$newExe) { throw 'Release archive contains no axiom_engine.exe' }
Get-Process axiom_engine -ErrorAction SilentlyContinue | Stop-Process -Force
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'axiom_engine_rs\target\release'), (Join-Path $Root 'checkpoints\memory') | Out-Null
Copy-Item -Force $newExe.FullName (Join-Path $Root 'axiom_engine_rs\target\release\axiom_engine.exe')
Copy-Item -Force (Join-Path $tmp 'axiom_bpe.json') (Join-Path $Root 'checkpoints\axiom_bpe.json')
Copy-Item -Force (Join-Path $tmp 'axiom_production_bpe.bin') (Join-Path $Root 'checkpoints\axiom_production_bpe.bin')
Copy-Item -Force (Join-Path $tmp 'axiom_production_bpe.meta.json') (Join-Path $Root 'checkpoints\axiom_production_bpe.meta.json')
Set-Content -NoNewline -Encoding ASCII -Path $VersionFile -Value $tag
& (Join-Path $PSScriptRoot 'restart_proxy.ps1')
