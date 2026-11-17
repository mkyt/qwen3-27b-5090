# Check the real checksum selector without running downloads, installers or the update entry point.
$ErrorActionPreference = "Stop"
$source = Join-Path $PSScriptRoot "..\update.ps1"
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $source), [ref]$tokens, [ref]$errors)
if ($errors) { throw "update.ps1 did not parse: $errors" }
$function = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                        $node.Name -eq "Expected-Hash" }, $true)
if (-not $function) { throw "Expected-Hash not found" }
Invoke-Expression $function.Extent.Text
$work = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString())
New-Item -ItemType Directory $work | Out-Null
$hash = "a" * 64
$zip = "qwen-test-windows-x64.zip"
$fixtures = @{}
function Download($url, $file) {
  if (-not $fixtures.ContainsKey($url)) { throw "simulated download failure" }
  Set-Content -Encoding ascii $file $fixtures[$url]
}
function Release($name, $url = "sum") { return @{ assets = @(@{ name = $name; url = $url }) } }
function Must-Fail($release) {
  try { $null = Expected-Hash $release $zip } catch {
    if ($_.Exception.Message -notlike "cannot verify the update:*") { throw }
    return
  }
  throw "unverifiable checksum was accepted"
}
try {
  Write-Host "Checking a named checksum"
  $fixtures["sum"] = "$hash  $zip"
  if ((Expected-Hash (Release "$zip.sha256") $zip) -ne $hash) { throw "named checksum rejected" }
  Write-Host "Checking a dedicated bare checksum"
  $fixtures["sum"] = $hash
  if ((Expected-Hash (Release "$zip.sha256") $zip) -ne $hash) { throw "dedicated bare checksum rejected" }
  Write-Host "Checking rejection of ambiguous, mismatched and unavailable checksums"
  Must-Fail (Release "SHA256SUMS")  # a bare hash in a general manifest cannot name the zip
  $fixtures["sum"] = "$hash  unrelated.zip"
  Must-Fail (Release "SHA256SUMS")  # even a single-line manifest must match the filename
  Must-Fail (Release "$zip.sha256" "offline")
  Must-Fail @{ assets = @() }
  Write-Host "Checking a multi-file manifest"
  $fixtures["sum"] = @(
    (("b" * 64) + "  unrelated.zip")
    "$hash  $zip"
  )
  if ((Expected-Hash (Release "SHA256SUMS") $zip) -ne $hash) { throw "manifest checksum rejected" }
  Write-Host "7 updater checksum checks passed"
} finally {
  Remove-Item -Recurse -Force $work
}
