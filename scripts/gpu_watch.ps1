# gpu_watch.ps1 — log per-card VRAM totals + docker container state, timestamped.
# Usage:
#   powershell -File gpu_watch.ps1            # one snapshot
#   powershell -File gpu_watch.ps1 -Loop 60   # every 60 s, Ctrl+C to stop
# Output: logs/gpu_watch.log (repo root) — diff lines to spot jumps.

param([int]$Loop = 0)

$logDir = Join-Path $PSScriptRoot "..\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir "gpu_watch.log"

function Get-Snapshot {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $lines = @("== $ts")
    # per-card totals (nvidia-smi; WDDM: per-process is [N/A], totals are reliable)
    $smi = & nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>$null
    foreach ($row in $smi) {
        $c = $row -split ',' | ForEach-Object { $_.Trim() }
        $util = ($c[4] -replace '[^0-9.]', '').Trim()
        $lines += ("  GPU{0} {1}: {2}/{3} MiB, {4}% util" -f $c[0], $c[1], $c[2], $c[3], $util)
    }
    # docker containers (running) with their image — attribution by diffing around stop/start
    $dock = & docker ps --format "{{.Names}}|{{.Image}}" 2>$null
    if ($dock) { foreach ($d in $dock) { $lines += "  docker: $d" } } else { $lines += "  docker: (none running)" }
    return ($lines -join "`n")
}

do {
    $snap = Get-Snapshot
    Add-Content -Path $logFile -Value $snap -Encoding utf8
    Write-Output $snap
    if ($Loop -gt 0) { Start-Sleep -Seconds $Loop }
} while ($Loop -gt 0)
