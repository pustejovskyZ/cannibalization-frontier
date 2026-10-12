# ============================================================================
# run_final_baselines.ps1 -- S.105 final pre-launch check of the EXACT rerun
# configuration (TYNDP_FIXES=all incl. fuel + hub-profile fix): 4 baselines +
# TYNDP battery benchmark, 12 weeks, ~20-25 min. Results -> logs\QUEUE_STATUS.txt
# ============================================================================
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$env:LP_SOLVER_THREADS = "2"; $env:GUROBI_METHOD = "2"; $env:SOLVE_TIME_LIMIT_S = "1800"
$env:CAP_OVERRIDES = ""; $env:STO_EP_RATIO = ""; $env:OFFSHORE_HUBS = ""; $env:GRID_STEP_MULT = ""; $env:H2_OVERRIDES = ""
$env:TYNDP_FIXES = "all"
$py = "python"
if (-not (Test-Path $py)) { $py = "python" }
New-Item -ItemType Directory -Force -Path logs | Out-Null
Add-Content logs\QUEUE_STATUS.txt "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  === run_final_baselines.ps1 START ==="
foreach ($sc in @("de2040","de2035","de2050","nt2030")) {
  $log = "logs\final_$sc.log"
  & $py directed_search.py --scenario $sc --climate-year 2009 --demand-2040 `
    --h2-2040 --ev-2040 --p2g-endogenous --n-weeks 12 --baseline-only 2>&1 | Tee-Object $log
  $line = (Select-String -Path $log -Pattern "B_opt cost" | Select-Object -Last 1).Line
  Add-Content logs\QUEUE_STATUS.txt "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  FINAL $sc : $line"
}
& $py tyndp_battery_benchmark.py --scenarios "nt2030,de2035,de2040,de2050" --out output\tyndp_battery_benchmark_final.csv 2>&1 | Tee-Object logs\final_battery_benchmark.log
$env:TYNDP_FIXES = ""
Add-Content logs\QUEUE_STATUS.txt "$(Get-Date -Format 'yyyy-MM-dd HH:mm')  === run_final_baselines.ps1 FINISHED ==="
& $py notify.py "run_final_baselines finished - check logs\QUEUE_STATUS.txt"
