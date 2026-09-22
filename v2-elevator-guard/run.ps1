# 启动电梯危险行为智能预警系统。
#
# 端口优先级： -Port 参数  >  环境变量 PORT  >  backend\.env 的 PORT  >  8001
# 解释器优先级： -Python 参数  >  环境变量 POSE_WEB_PYTHON  >  常见 conda 路径  >  PATH 里的 python
#
# 用法：
#   .\run.ps1
#   .\run.ps1 -Port 8020
#   .\run.ps1 -Python "C:\path\to\python.exe"
#   $env:PORT=8020; .\run.ps1
param(
    [int]$Port = 0,
    [string]$Python = "",
    [switch]$NoReload
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location "$root\backend"

# ---- 选解释器 ----------------------------------------------------------
function Resolve-Python {
    param([string]$explicit)
    if ($explicit) {
        if (-not (Test-Path $explicit)) { Write-Error "指定的解释器不存在：$explicit" }
        return $explicit
    }
    if ($env:POSE_WEB_PYTHON) {
        if (-not (Test-Path $env:POSE_WEB_PYTHON)) {
            Write-Error "环境变量 POSE_WEB_PYTHON 指向的解释器不存在：$($env:POSE_WEB_PYTHON)"
        }
        return $env:POSE_WEB_PYTHON
    }
    foreach ($candidate in @(
        'D:\Conda\envs\poseweb2\python.exe',
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe"
    )) {
        if (Test-Path $candidate) { return $candidate }
    }
    # 退回到 PATH 里的 python；找不到就直接报错，别让 uvicorn 抛一堆栈
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    Write-Error "没找到可用的 Python 解释器。请用 -Python 参数指定，或设置环境变量 POSE_WEB_PYTHON。"
}

$pythonExe = Resolve-Python $Python

# ---- 选端口 ------------------------------------------------------------
if ($Port -eq 0 -and $env:PORT) { $Port = [int]$env:PORT }
if ($Port -eq 0 -and (Test-Path ".env")) {
    $line = Select-String -Path ".env" -Pattern "^\s*PORT\s*=\s*(\d+)" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($line) { $Port = [int]$line.Matches[0].Groups[1].Value }
}
if ($Port -eq 0) { $Port = 8001 }

# 实际试绑端口判断是否可用；比 Get-NetTCPConnection 可靠
# （后者在部分受限账户下会静默返回空结果，导致误判为“端口空闲”）。
function Test-PortFree([int]$p) {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

if (-not (Test-PortFree $Port)) {
    Write-Host "端口 $Port 已被占用。" -ForegroundColor Red
    $busy = netstat -ano | Select-String ":$Port\s" | Select-String "LISTENING" | Select-Object -First 1
    if ($busy) {
        $ownerPid = ($busy.ToString().Trim() -split '\s+')[-1]
        $owner = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
        if ($owner) {
            Write-Host ("  占用进程：{0} (PID {1})" -f $owner.ProcessName, $ownerPid) -ForegroundColor Red
        }
    }
    $suggestion = $Port + 1
    while ($suggestion -lt $Port + 50 -and -not (Test-PortFree $suggestion)) { $suggestion++ }
    Write-Host ""
    Write-Host "换个端口即可，例如：" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 -Port $suggestion" -ForegroundColor Yellow
    exit 1
}

$env:DEVICE = if ($env:DEVICE) { $env:DEVICE } else { 'cuda' }

$uvicornArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
if (-not $NoReload) { $uvicornArgs += '--reload' }

Write-Host "启动服务：http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "接口文档：http://127.0.0.1:$Port/docs" -ForegroundColor DarkGray
Write-Host "解释器　：$pythonExe" -ForegroundColor DarkGray
Write-Host "推理设备：$env:DEVICE" -ForegroundColor DarkGray
& $pythonExe @uvicornArgs
