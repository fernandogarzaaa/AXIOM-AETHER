param(
    [string]$Repo = $(if ($env:AXIOM_REPO) { $env:AXIOM_REPO } else { "fernandogarzaaa/AXIOM-AETHER" }),
    [string]$InstallDir = $(if ($env:AXIOM_INSTALL_DIR) { $env:AXIOM_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Axiom\bin" })
)

$ErrorActionPreference = "Stop"
$apiUrl = "https://api.github.com/repos/$Repo/releases/latest"

Write-Host "axiom installer: resolving latest release from $Repo"
$release = Invoke-RestMethod -Uri $apiUrl
$asset = $release.assets | Where-Object {
    $_.name -match '^axiom-ttt-.*-windows-x86_64\.zip$'
} | Select-Object -First 1

if (-not $asset) {
    throw "axiom installer: no windows-x86_64 release asset found"
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("axiom-install-" + [System.Guid]::NewGuid())
$archive = Join-Path $tmp "axiom.zip"
$extract = Join-Path $tmp "extract"
New-Item -ItemType Directory -Path $tmp, $extract | Out-Null

try {
    Write-Host "axiom installer: downloading $($asset.browser_download_url)"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive
    Expand-Archive -Path $archive -DestinationPath $extract -Force
    $binary = Get-ChildItem -Path $extract -Recurse -Filter "axiom_engine.exe" | Select-Object -First 1
    if (-not $binary) {
        throw "axiom installer: release archive did not contain axiom_engine.exe"
    }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $target = Join-Path $InstallDir "axiom.exe"
    Copy-Item -Path $binary.FullName -Destination $target -Force

    Write-Host "axiom installer: installed $target"
    $pathEntries = $env:PATH -split ';'
    if ($pathEntries -notcontains $InstallDir) {
        Write-Host "axiom installer: add this directory to PATH: $InstallDir"
    }

    & $target init
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
