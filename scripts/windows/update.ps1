<#
Updates the Windows package (app\scripts\update.ps1; update.cmd runs a copy of it) to the latest release of
mkyt/qwen3-27b-5090, or to the one -Tag names (an older one too). settings.ini, the weights and the log stay.

Installed packages run their own copy of this script to install later releases, so it assumes as little as it can
about them. This copy finds the release, downloads its zip (curl.exe; an interrupted download resumes), checks it
against the release's SHA-256 and unpacks it, then stops what it knows runs from this folder (its own tray app, as the
tray's Quit does, and the server). The new package then installs itself: its own update.ps1, from a copy in %TEMP%,
with -InstallFrom (the unpacked package), -Root and -StartTray; it knows its layout and its tray app. Only when the new
package cannot (no update.ps1, or it stops before changing anything) does this copy install it: every entry at the
top of the new package replaces the entry of that name here (folders whole, files overwritten), except the user's
data (settings.ini, huggingface\, logs\, update\), and everything goes back if a step fails.

What later releases keep, for the installed copies of this script:
- The latest release of the repository (not a draft or pre-release) is the one to install; try-out versions are
  pre-releases. Its tag equals the new package's app\VERSION (only equality is compared; no VERSION: always updated).
- An asset named *-windows-x64.zip (else the first *windows*.zip), and its SHA-256 in <zip name>.sha256 or in a
  SHA256SUMS-style asset ("<hash>  <name>" lines). Without either, the zip is installed unchecked.
- The zip holds the package, at its top or in a folder: the folder that has python\python.exe.
- The new package's update.ps1 accepts -InstallFrom, -Root, -StartTray and -Started (and -Check, which it ignores
  then), and creates the file -Started names before it changes anything. Anything else (where things are, the tray
  app, the server) may change.

    update.cmd [-Tag v0.3.1] [-Check] [-Force] [-Zip <a downloaded zip>]
#>
[CmdletBinding()]  # (an argument it does not know is an error, not ignored)
param(
  [string]$Tag = "",
  [switch]$Check,  # only print the installed and the release's version
  [switch]$Force,  # also when the version is the same
  [string]$Zip = "",  # a zip already downloaded (its SHA-256 is not checked)
  [string]$Repo = "mkyt/qwen3-27b-5090",
  [string]$Root = "",  # the package's folder (update.cmd runs a copy of this script from %TEMP%)
  # The install step, which an older copy of this script hands over: the unpacked package of this script
  [string]$InstallFrom = "",
  [switch]$StartTray,
  [string]$Started = ""
)
$ErrorActionPreference = "Stop"
# (Run from a copy: PowerShell keeps the script open, and app\ could not be moved aside with it in there.)
$root = (Resolve-Path $(if ($Root) { $Root } else { Join-Path $PSScriptRoot "..\.." })).Path
$work = Join-Path $root "update"
$userData = "settings.ini", "huggingface", "logs", "update"  # (never replaced)
$agent = @{ "User-Agent" = "qwen3-27b-5090-update" }

function Get-Version($dir) {
  $f = Join-Path $dir "app\VERSION"
  if (Test-Path $f) { (Get-Content $f -TotalCount 1).Trim() } else { "" }
}

function Retry([scriptblock]$op) {  # (a virus scanner may hold a new file for a moment)
  for ($i = 0; ; $i++) {
    try { & $op; return } catch { if ($i -ge 10) { throw }; Start-Sleep -Milliseconds 500 }
  }
}

function Move-Entry($from, $to) {
  # (Directory.Move renames the folder or fails as a whole; Move-Item moves what it can file by file)
  if (Test-Path $from -PathType Container) { Retry { [IO.Directory]::Move($from, $to) } } else { Retry { [IO.File]::Move($from, $to) } }
}

# Each entry at the top of $pkg replaces the one of that name in $root; all goes back if a step fails
function Install-Package($pkg, [bool]$startTray) {
  $old = Join-Path $work "old"
  if (Test-Path $old) { Remove-Item -Recurse -Force $old }
  New-Item -ItemType Directory -Force $old | Out-Null
  $done = @()  # (name, whether there was one before)
  try {
    foreach ($e in Get-ChildItem -Force $pkg) {
      if ($userData -contains $e.Name) { Write-Host "(not replacing $($e.Name): user data)"; continue }
      $dst = Join-Path $root $e.Name
      $had = Test-Path $dst
      if ($had) { Move-Entry $dst (Join-Path $old $e.Name) }
      $done += , @($e.Name, $had)
      Move-Entry $e.FullName $dst
    }
  } catch {
    $err = $_
    Write-Host "The update failed: putting the old version back."
    [array]::Reverse($done)
    foreach ($d in $done) {
      $dst = Join-Path $root $d[0]
      if (Test-Path $dst) { Move-Entry $dst (Join-Path $pkg $d[0]) }
      if ($d[1]) { Move-Entry (Join-Path $old $d[0]) $dst }
    }
    Write-Host "Nothing changed. If a file of the package is in use (an editor, a terminal in its folder, a virus scan),"
    Write-Host "close it and run update.cmd again."
    throw $err
  }
  if ($startTray) {
    $tray = Join-Path $root "qwen3-27b-tray.exe"
    if (Test-Path $tray) { Start-Process $tray; Write-Host "Started the tray app again." }
  }
}

if ($InstallFrom) {  # (the step an older copy hands over)
  if ($Started) { New-Item -ItemType File -Force $Started | Out-Null }
  Install-Package (Resolve-Path $InstallFrom).Path $StartTray.IsPresent
  exit 0
}

$installed = Get-Version $root
$shown = if ($installed) { $installed } else { "unknown (before v0.3.1)" }
if (-not (Test-Path (Join-Path $root "python\python.exe"))) {
  throw "$root is not the folder of the Windows package (python\python.exe not found)"
}

function Download($url, $file, [switch]$Resume) {  # (-Resume: the zip, with progress; else a small file, quietly)
  $curl = Join-Path $env:SystemRoot "System32\curl.exe"
  if (Test-Path $curl) {
    $more = if ($Resume) { @("-C", "-") } else { @("-s", "-S") }
    & $curl -L --fail --retry 5 --retry-delay 3 @more -o $file $url
    if ($LASTEXITCODE) { throw "download failed (curl exit code $LASTEXITCODE); run update.cmd again to resume" }
  } else {
    Invoke-WebRequest -UseBasicParsing -Headers $agent $url -OutFile $file
  }
}

function Find-Release {
  [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
  $api = "https://api.github.com/repos/$Repo/releases/" + $(if ($Tag) { "tags/$Tag" } else { "latest" })
  try {
    $r = Invoke-RestMethod $api -Headers ($agent + @{ "Accept" = "application/vnd.github+json" })
    return @{ tag = $r.tag_name; assets = @($r.assets | ForEach-Object { @{ name = $_.name; url = $_.browser_download_url; size = $_.size } }) }
  } catch {
    # (no API, e.g. over its rate limit: the release page's redirect names the tag, and the asset has the usual name)
    Write-Host "(GitHub API: $($_.Exception.Message); trying the release page)"
    $t = $Tag
    if (-not $t) {
      $req = [Net.WebRequest]::Create("https://github.com/$Repo/releases/latest")
      $req.AllowAutoRedirect = $false
      $resp = $req.GetResponse()
      $t = ($resp.Headers["Location"] -split "/")[-1]
      $resp.Close()
    }
    if (-not $t -or $t -eq "releases") { throw "cannot find the latest release" }
    $name = "qwen3-27b-5090-$t-windows-x64.zip"
    $base = "https://github.com/$Repo/releases/download/$t"
    return @{ tag = $t; assets = @(@{ name = $name; url = "$base/$name"; size = 0 }, @{ name = "$name.sha256"; url = "$base/$name.sha256"; size = 0 }) }
  }
}

function Expected-Hash($release, $zipName) {
  $sums = @($release.assets | Where-Object { $_.name -eq "$zipName.sha256" }) +
          @($release.assets | Where-Object { $_.name -ne "$zipName.sha256" -and $_.name -match "sha256" })
  foreach ($a in $sums) {
    $f = Join-Path $work "sums.txt"
    try { Download $a.url $f } catch { continue }
    $lines = @(Get-Content $f | Where-Object { $_.Trim() })
    Remove-Item $f
    foreach ($l in $lines) {
      $parts = $l.Trim() -split "\s+", 2
      $file = if ($parts.Count -gt 1) { $parts[1].TrimStart("*").Trim() } else { "" }
      if ($parts[0] -match "^[0-9a-fA-F]{64}$" -and ($lines.Count -eq 1 -or $file -eq $zipName)) { return $parts[0].ToLower() }
    }
  }
  return $null
}

# The release and its zip
if (-not $Zip) {
  $release = Find-Release
  $asset = @($release.assets | Where-Object { $_.name -like "*-windows-x64.zip" }) +
           @($release.assets | Where-Object { $_.name -like "*windows*.zip" }) | Select-Object -First 1
  if (-not $asset) { throw "release $($release.tag) has no Windows package" }
  $size = if ($asset.size) { " ($([math]::Round($asset.size / 1MB, 1)) MB)" } else { "" }
  Write-Host "installed: $shown, release: $($release.tag)$size"
  if ($Check) { exit 0 }
  if ($installed -eq $release.tag -and -not $Force) {
    Write-Host "Already up to date (-Force reinstalls it)."
    exit 0
  }
  New-Item -ItemType Directory -Force $work | Out-Null
  $Zip = Join-Path $work $asset.name
  $want = Expected-Hash $release $asset.name
  $have = if ($want -and (Test-Path $Zip)) { (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower() } else { "" }
  if (-not $want -or $have -ne $want) {
    Download $asset.url $Zip -Resume
    if ($want -and (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLower() -ne $want) {
      Remove-Item $Zip
      throw "the download's SHA-256 does not match the release's; run update.cmd again"
    }
  }
  if (-not $want) { Write-Host "(the release has no SHA-256 for it: not checked)" }
} else {
  $Zip = (Resolve-Path $Zip).Path
  Write-Host "installed: $shown, zip: $Zip"
  if ($Check) { exit 0 }
  New-Item -ItemType Directory -Force $work | Out-Null
}

# Unpack next to the package (the same volume: installing only renames)
$stage = Join-Path $work "new"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory $stage | Out-Null
$tar = Join-Path $env:SystemRoot "System32\tar.exe"
if (Test-Path $tar) {
  & $tar -x -f $Zip -C $stage
  if ($LASTEXITCODE) { throw "cannot unpack $Zip" }
} else {
  Expand-Archive $Zip $stage
}
$pkg = @(@(Get-Item $stage) + @(Get-ChildItem -Directory $stage) + @(Get-ChildItem -Directory $stage | Get-ChildItem -Directory) |
  Where-Object { Test-Path (Join-Path $_.FullName "python\python.exe") }) | Select-Object -First 1
if (-not $pkg) { throw "$Zip does not hold the Windows package (no python\python.exe)" }
$pkg = $pkg.FullName
$newVersion = Get-Version $pkg

# Stop what runs from this folder, as this version knows it: its tray app as the Quit menu item does (WM_COMMAND,
# ID_QUIT = 6 in windows\tray\tray.cc), which stops the server, and a server of start.cmd by hand
Add-Type -Namespace Q27 -Name Win -MemberDefinition @'
[DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowW(string cls, IntPtr title);
[DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr window, uint message, IntPtr wparam, IntPtr lparam);
'@
function Running($exe) {
  @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path -eq $exe })
}
$tray = Join-Path $root "qwen3-27b-tray.exe"
$trayWasRunning = (Running $tray).Count -gt 0
if ($trayWasRunning) {
  Write-Host "Quitting the tray app (it stops the server)..."
  $window = [Q27.Win]::FindWindowW("qwen3-27b-5090-tray", [IntPtr]::Zero)  # (a $null string would be "")
  if ($window -ne [IntPtr]::Zero) { [Q27.Win]::PostMessageW($window, 0x0111, [IntPtr]6, [IntPtr]::Zero) | Out-Null }
  $t0 = Get-Date
  $asked = $false
  while ((Running $tray).Count -gt 0) {
    if (-not $asked -and ((Get-Date) - $t0).TotalSeconds -gt 60) {
      Write-Host "The tray app is still running: quit it from its menu (Quit). Waiting..."
      $asked = $true
    }
    Start-Sleep -Milliseconds 500
  }
}
$python = Join-Path $root "python\python.exe"
if ((Running $python).Count -gt 0) {
  Write-Host "The server of this folder is running (start.cmd): close its window. Waiting..."
  while ((Running $python).Count -gt 0) { Start-Sleep -Seconds 1 }
}

# The new package installs itself if it can; this copy installs it otherwise
$installer = Join-Path $pkg "app\scripts\update.ps1"
$handed = $false
try {
  if (Test-Path $installer) {
    $copy = Join-Path $env:TEMP "qwen3-27b-5090-install.ps1"
    $mark = Join-Path $work "install-started"
    Copy-Item -Force $installer $copy
    if (Test-Path $mark) { Remove-Item $mark }
    # (-Check too: a script that ignored -InstallFrom would only print the versions, not update this folder itself)
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $copy, "-InstallFrom", $pkg, "-Root", $root, "-Started", $mark, "-Check")
    if ($trayWasRunning) { $argList += "-StartTray" }  # (-File takes no -Switch:$false)
    & powershell @argList
    $code = $LASTEXITCODE
    if (Test-Path $mark) {
      $handed = $true
      if ($code) { throw "the new version's installer failed (exit code $code)" }
    } else {
      Write-Host "(the new version's installer did not start: installing it with this one)"
    }
  }
  if (-not $handed) { Install-Package $pkg $trayWasRunning }
} catch {
  if ($trayWasRunning -and (Test-Path $tray)) { Start-Process $tray; Write-Host "Started the tray app again." }  # (the old one)
  throw
}
try { Remove-Item -Recurse -Force $work } catch { Write-Host "(could not delete $work; delete it later)" }
$now = Get-Version $root
Write-Host ("Updated: $shown -> " + $(if ($now) { $now } elseif ($newVersion) { $newVersion } else { "unknown" }))
