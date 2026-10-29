<#
Builds the Windows package dist\qwen3-27b-5090-windows-x64.zip: q27.dll, the embeddable Python (python.org) with the
server's packages, the server and start.cmd (scripts\windows\start.cmd). Like the serving image, it holds no compiler,
CUDA toolkit or PyTorch: the GPU driver is the only requirement.

Run on Windows in a shell with MSVC and the CUDA toolkit (vcvars64.bat, then CUDA_PATH and CUDACXX), CMake and Ninja
(those of Visual Studio do), and a CPython with pip, whose pip downloads the wheels for the package's Python:
    powershell -ExecutionPolicy Bypass -File scripts\windows\package.ps1 [-Python python] [-PythonVersion 3.14.10]
#>
param(
  [string]$Python = "python",
  [string]$PythonVersion = "3.14.10",
  [string]$Out = "dist"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$name = "qwen3-27b-5090-windows-x64"
$dst = Join-Path $Out $name
$ver = $PythonVersion.Split(".")
$tag = $ver[0] + $ver[1]

# The engine, where q27lib looks for it (engine\build\q27.dll)
cmake -S engine -B engine\build -G Ninja -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE) { throw "cmake configure failed" }
cmake --build engine\build --target q27
if ($LASTEXITCODE) { throw "cmake build failed" }

if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
New-Item -ItemType Directory -Force $dst | Out-Null

# Python: the embeddable package (signed by the PSF; unsigned interpreters meet SmartScreen), the server's packages in
# Lib\site-packages. Its ._pth file replaces sys.path: the standard library zip, its directory and the packages (the
# server adds its own directories).
$embed = Join-Path $Out "python-$PythonVersion-embed-amd64.zip"
if (-not (Test-Path $embed)) {
  Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" -OutFile $embed
}
Expand-Archive $embed (Join-Path $dst "python")
Set-Content -Encoding ascii (Join-Path $dst "python\python$tag._pth") "python$tag.zip`r`n.`r`nLib\site-packages"
$site = Join-Path $dst "python\Lib\site-packages"
& $Python -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --platform win_amd64 `
  --implementation cp --python-version "$($ver[0]).$($ver[1])" --target $site -r frontend\requirements.txt
if ($LASTEXITCODE) { throw "pip install failed" }
if (Test-Path "$site\bin") { Remove-Item -Recurse -Force "$site\bin" }  # (numpy's f2py launcher)

# The server: what the serving image copies (Dockerfile), the weight download and q27.dll
$app = Join-Path $dst "app"
foreach ($d in "bench\draft-vocab", "engine\python", "frontend") {
  New-Item -ItemType Directory -Force (Split-Path (Join-Path $app $d)) | Out-Null
  Copy-Item -Recurse $d (Join-Path $app $d)
}
Remove-Item -Recurse -Force (Join-Path $app "frontend\tests")
New-Item -ItemType Directory -Force "$app\reference\qwen27b", "$app\scripts", "$app\engine\build" | Out-Null
foreach ($f in "__init__.py", "paths.py", "image_tokens.py") { Copy-Item "reference\qwen27b\$f" "$app\reference\qwen27b\" }
Copy-Item scripts\download_weights.py "$app\scripts\"
Copy-Item engine\build\q27.dll "$app\engine\build\"
Get-ChildItem -Recurse -Directory -Filter __pycache__ $app | Remove-Item -Recurse -Force
Copy-Item LICENSE $dst
Copy-Item scripts\windows\start.cmd (Join-Path $dst "start.cmd")

$zip = Join-Path $Out "$name.zip"
if (Test-Path $zip) { Remove-Item $zip }
# (Windows' bsdtar, by its path: Git's GNU tar may come first in PATH and makes no zip. A zip with forward
# slashes, unlike Compress-Archive in PowerShell 5.1.)
& "$env:SystemRoot\System32\tar.exe" -a -c -f $zip -C $Out $name
if ($LASTEXITCODE) { throw "zip failed" }
Write-Host ("{0}: {1:N1} MB" -f $zip, ((Get-Item $zip).Length / 1MB))
