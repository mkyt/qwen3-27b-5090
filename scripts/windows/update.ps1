<#
Updates the Windows package (app\scripts\update.ps1; update.cmd runs a copy of it) to the latest release of
mkyt/qwen3-27b-5090, or to the one -Tag names (an older one too). settings.ini, the weights and the log stay.

The zip downloads into update\ with curl.exe (an interrupted download resumes) and is checked against the release's
SHA-256. The tray app of this folder is quit as its menu's Quit does and started again afterwards; a server of
start.cmd has to be closed by hand. python\ and app\ are moved aside, the new ones moved in, and moved back if that
fails. The package's version is app\VERSION (none before v0.3.1: always updated).

    update.cmd [-Tag v0.3.1] [-Check] [-Force] [-Zip <a downloaded zip>]
#>
param(
  [string]$Tag = "",
  [switch]$Check,  # only print the installed and the release's version
  [switch]$Force,  # also when the version is the same
  [string]$Zip = "",  # a zip already downloaded (its SHA-256 is not checked)
  [string]$Repo = "mkyt/qwen3-27b-5090",
  [string]$Root = ""  # the package's folder (update.cmd runs a copy of this script from %TEMP%)
)
$ErrorActionPreference = "Stop"
# (Run from a copy: PowerShell keeps the script open, and app\ could not be moved aside with it in there.)
$root = (Resolve-Path $(if ($Root) { $Root } else { Join-Path $PSScriptRoot "..\.." })).Path
$tray = Join-Path $root "qwen3-27b-tray.exe"
$work = Join-Path $root "update"
$versionFile = Join-Path $root "app\VERSION"
$installed = if (Test-Path $versionFile) { (Get-Content $versionFile -TotalCount 1).Trim() } else { "" }
$shown = if ($installed) { $installed } else { "unknown (before v0.3.1)" }
if (-not ((Test-Path (Join-Path $root "python\python.exe")) -and (Test-Path $tray))) {
  throw "$root is not the folder of the Windows package (python\python.exe and qwen3-27b-tray.exe not found)"
}

# The release and its assets
if (-not $Zip) {
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  $api = "https://api.github.com/repos/$Repo/releases/" + $(if ($Tag) { "tags/$Tag" } else { "latest" })
  $release = Invoke-RestMethod $api -Headers @{ "User-Agent" = "qwen3-27b-5090-update"; "Accept" = "application/vnd.github+json" }
  $asset = $release.assets | Where-Object { $_.name -like "*-windows-x64.zip" } | Select-Object -First 1
  $sumAsset = $release.assets | Where-Object { $asset -and $_.name -eq "$($asset.name).sha256" } | Select-Object -First 1
  if (-not $asset) { throw "release $($release.tag_name) has no Windows package" }
  $Tag = $release.tag_name
  Write-Host "installed: $shown, release: $Tag ($([math]::Round($asset.size / 1MB, 1)) MB)"
  if ($Check) { exit 0 }
  if ($installed -eq $Tag -and -not $Force) {
    Write-Host "Already up to date (-Force reinstalls it)."
    exit 0
  }
  New-Item -ItemType Directory -Force $work | Out-Null
  $Zip = Join-Path $work $asset.name
  $want = $null
  if ($sumAsset) {
    $sumText = & "$env:SystemRoot\System32\curl.exe" -L --fail -s $sumAsset.browser_download_url
    if ($LASTEXITCODE) { throw "cannot download $($sumAsset.name) (curl exit code $LASTEXITCODE)" }
    $want = ([string]$sumText).Trim().Split(" ")[0].ToLower()
  }
  $have = if (Test-Path $Zip) { (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower() } else { "" }
  if (-not $want -or $have -ne $want) {
    # (curl.exe of Windows; -C - resumes a partial file)
    & "$env:SystemRoot\System32\curl.exe" -L --fail --retry 5 --retry-delay 3 -C - -o $Zip $asset.browser_download_url
    if ($LASTEXITCODE) { throw "download failed (curl exit code $LASTEXITCODE); run update.cmd again to resume" }
    if ($want) {
      $have = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower()
      if ($have -ne $want) {
        Remove-Item $Zip
        throw "the download's SHA-256 does not match the release's; run update.cmd again"
      }
    }
  }
  if (-not $want) { Write-Host "(the release has no .sha256: not checked)" }
} else {
  $Zip = (Resolve-Path $Zip).Path
  Write-Host "installed: $shown, zip: $Zip"
  if ($Check) { exit 0 }
  New-Item -ItemType Directory -Force $work | Out-Null
}

# Unpack next to the package (the same volume: the swap below only renames)
$stage = Join-Path $work "new"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory $stage | Out-Null
& "$env:SystemRoot\System32\tar.exe" -x -f $Zip -C $stage
if ($LASTEXITCODE) { throw "cannot unpack $Zip" }
$new = Get-ChildItem -Directory $stage | Where-Object { Test-Path (Join-Path $_.FullName "app\scripts\start_package.py") } |
  Select-Object -First 1
if (-not $new) { throw "$Zip does not hold the Windows package" }
$new = $new.FullName

# Stop what runs from this folder: the tray app as its Quit does (WM_COMMAND, ID_QUIT = 6), which stops its server
Add-Type -Namespace Q27 -Name Win -MemberDefinition @'
[DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowW(string cls, IntPtr title);
[DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr window, uint message, IntPtr wparam, IntPtr lparam);
'@
function Running($exe) {
  @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path -eq $exe })
}
$trayWasRunning = (Running $tray).Count -gt 0
if ($trayWasRunning) {
  Write-Host "Quitting the tray app (it stops the server)..."
  $window = [Q27.Win]::FindWindowW("qwen3-27b-5090-tray", [IntPtr]::Zero)  # (a $null string would be "")
  if ($window -ne [IntPtr]::Zero) { [Q27.Win]::PostMessageW($window, 0x0111, [IntPtr]6, [IntPtr]::Zero) | Out-Null }
  $t0 = Get-Date
  while ((Running $tray).Count -gt 0) {
    if (((Get-Date) - $t0).TotalSeconds -gt 120) { throw "the tray app did not quit; quit it from its menu and run update.cmd again" }
    Start-Sleep -Milliseconds 500
  }
}
$python = Join-Path $root "python\python.exe"
if ((Running $python).Count -gt 0) {
  Write-Host "The server of this folder is running (start.cmd): close its window. Waiting..."
  while ((Running $python).Count -gt 0) { Start-Sleep -Seconds 1 }
}

# Swap: the old python\ and app\ aside, the new ones in, then the files at the top (this script is loaded already).
# Everything goes back if a step fails.
$old = Join-Path $work "old"
if (Test-Path $old) { Remove-Item -Recurse -Force $old }
New-Item -ItemType Directory (Join-Path $old "top") | Out-Null
function Retry([scriptblock]$op) {  # (a virus scanner may hold a new file for a moment)
  for ($i = 0; ; $i++) {
    try { & $op; return } catch { if ($i -ge 10) { throw }; Start-Sleep -Milliseconds 500 }
  }
}
$aside, $placed, $files = @(), @(), @()
try {
  foreach ($d in "python", "app") {
    # (Directory.Move renames the folder or fails as a whole; Move-Item moves what it can file by file)
    Retry { [IO.Directory]::Move((Join-Path $root $d), (Join-Path $old $d)) }
    $aside += $d
    Retry { [IO.Directory]::Move((Join-Path $new $d), (Join-Path $root $d)) }
    $placed += $d
  }
  foreach ($f in Get-ChildItem -File $new) {
    $dst = Join-Path $root $f.Name
    if (Test-Path $dst) { Copy-Item $dst (Join-Path $old "top") }
    $files += $f.Name
    Retry { Copy-Item -Force $f.FullName $dst }
  }
} catch {
  $err = $_
  Write-Host "The update failed: putting the old version back."
  foreach ($name in $files) {
    $saved = Join-Path $old "top\$name"
    if (Test-Path $saved) { Copy-Item -Force $saved $root } else { Remove-Item -Force (Join-Path $root $name) -ErrorAction SilentlyContinue }
  }
  foreach ($d in $placed) { [IO.Directory]::Move((Join-Path $root $d), (Join-Path $new $d)) }
  foreach ($d in $aside) { [IO.Directory]::Move((Join-Path $old $d), (Join-Path $root $d)) }
  if ($trayWasRunning) { Start-Process $tray }
  Write-Host "Nothing changed. If a file in python\ or app\ is in use (an editor, a terminal in that folder, a virus scan),"
  Write-Host "close it and run update.cmd again."
  throw $err
}
try { Remove-Item -Recurse -Force $work } catch { Write-Host "(could not delete $work; delete it later)" }
$now = if (Test-Path $versionFile) { (Get-Content $versionFile -TotalCount 1).Trim() } else { "unknown" }
Write-Host "Updated: $shown -> $now"
if ($trayWasRunning) {
  Start-Process $tray
  Write-Host "Started the tray app again."
}
