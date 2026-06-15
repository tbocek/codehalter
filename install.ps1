#Requires -Version 5
<#
  codehalter installer for Windows.

    irm https://raw.githubusercontent.com/tbocek/codehalter/main/install.ps1 | iex

  Downloads the prebuilt codehalter.exe from the latest GitHub release, installs
  it to %LOCALAPPDATA%\Programs\codehalter, and adds that directory to your user
  PATH. The POSIX equivalent (Linux/macOS, or Windows under Git Bash/WSL) is
  install.sh.
#>
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 defaults can break the GitHub calls: force TLS 1.2 (the
# API rejects older) and silence the progress bar (it makes Invoke-WebRequest
# crawl on big downloads). Both are no-ops on PowerShell 7+.
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch {}

$Repo   = 'tbocek/codehalter'
$Binary = 'codehalter.exe'

# Only windows-amd64 is published today; bail clearly on anything else.
if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    Write-Error "Unsupported architecture: $env:PROCESSOR_ARCHITECTURE. Only windows-amd64 is published — see https://github.com/$Repo/releases/latest"
    return
}
$asset = 'codehalter-windows-amd64.exe'

# GitHub's API rejects requests without a User-Agent.
$ua = @{ 'User-Agent' = 'codehalter-install' }

Write-Host '==> Resolving latest codehalter release...'
$tag = (Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers $ua).tag_name
if (-not $tag) { Write-Error 'Failed to resolve the latest release tag.'; return }
Write-Host "==> Latest release: $tag"

$url  = "https://github.com/$Repo/releases/download/$tag/$asset"
$dir  = Join-Path $env:LOCALAPPDATA 'Programs\codehalter'
$dest = Join-Path $dir $Binary
New-Item -ItemType Directory -Force -Path $dir | Out-Null

Write-Host "==> Downloading $asset ..."
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
Write-Host "==> Installed: $dest"

# Add the install dir to the user PATH if it isn't already there.
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ([string]::IsNullOrEmpty($userPath)) {
    $newPath = $dir
} elseif ($userPath.Split(';') -notcontains $dir) {
    $newPath = "$userPath;$dir"
} else {
    $newPath = $null
}
if ($newPath) {
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Host "==> Added $dir to your user PATH (restart your shell to pick it up)."
}

Write-Host '==> Done. Point your editor''s ACP agent at the path above.'
Write-Host '==> Config & next steps: https://github.com/tbocek/codehalter#readme'
