# ============================================================================
# run_rerun_phase1.ps1 -- S.103/S.105 FULL RERUN, phase 1 (all TYNDP fixes on).
# USAGE (two parallel windows on the 8-core machine):
#   window 1:  .\run_rerun_phase1.ps1 -Queue A
#   window 2:  .\run_rerun_phase1.ps1 -Queue B
# CRASH / RESTART: just run the same command again. Finished runs are skipped,
#   an interrupted frontier resumes from its last finished target, finished
#   CMA restarts of the interrupted target are reloaded from restart_cache_*,
#   and an interrupted grid continues from its last solved combination.
# Files carry the suffix _tfall -> the old results are never touched.
# Telegram: every finished target reports queue position and ETAs.
# ============================================================================
param([ValidateSet("A","B","all")][string]$Queue = "all")
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$thr = if ($Queue -eq "all") { "2" } else { "1" }   # 2 queues x 4 workers x 1 thread = 8 cores
$env:LP_SOLVER_THREADS  = $thr; $env:GUROBI_METHOD = "2"; $env:SOLVE_TIME_LIMIT_S = "1800"
$env:CAP_OVERRIDES = ""; $env:STO_EP_RATIO = ""; $env:OFFSHORE_HUBS = ""; $env:GRID_STEP_MULT = ""; $env:H2_OVERRIDES = ""
$env:TYNDP_FIXES = "all"
$py = "python"
if (-not (Test-Path $py)) { $py = "python" }
$outdir = Join-Path $PSScriptRoot "output"; $logdir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$status = Join-Path $logdir "QUEUE_STATUS_$Queue.txt"   # one file per queue (no write clashes)
function Log($m) { $l = "{0}  [{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $Queue, $m; Write-Host $l; try { Add-Content -Path $status -Value $l } catch {} }

# ---- queue definition: kind, scenario, target, grid multiplier, expected size
# expected targets: previous campaign +1, raised for the vf runs because the new
# baselines start lower (VF_min 0.51-0.56 vs 0.59-0.65) -> longer frontiers
$A = @(
  @{k="f"; sc="de2040"; tg="vf";  m="1";   exp=9},
  @{k="f"; sc="de2050"; tg="vf";  m="1";   exp=8},
  @{k="f"; sc="nt2030"; tg="vf";  m="1";   exp=11},
  @{k="g"; sc="de2040"; exph=2},
  @{k="f"; sc="de2050"; tg="vfw"; m="1";   exp=4}
)
$B = @(
  @{k="f"; sc="de2040"; tg="vfw"; m="0.5"; exp=8},
  @{k="f"; sc="de2035"; tg="vf";  m="1";   exp=9},
  @{k="g"; sc="de2035"; exph=2},
  @{k="g"; sc="de2050"; exph=2},
  @{k="f"; sc="de2035"; tg="vfw"; m="1";   exp=5},
  @{k="f"; sc="nt2030"; tg="vfw"; m="1";   exp=6}
)
$items = if ($Queue -eq "A") { $A } elseif ($Queue -eq "B") { $B } else { $A + $B }

# progress plan for queue_progress.py (Telegram ETA)
$plan = @{ queue = $Queue; items = @() }
foreach ($it in $items) {
  if ($it.k -eq "f") { $plan.items += @{ label = "$($it.sc.ToUpper()) $($it.tg)"; kind = "frontier"; exp_targets = $it.exp } }
  else { $plan.items += @{ label = "$($it.sc.ToUpper()) grid"; kind = "grid"; exp_hours = $it.exph } }
}
$plan | ConvertTo-Json -Depth 4 | Set-Content -Encoding ascii (Join-Path $logdir "queue_$($Queue)_plan.json")
$env:QUEUE_NAME = $Queue

function Run-Frontier($i, $sc, $tg, $mult) {
  $rid  = "5node_${tg}_${sc}_cy2009_dhep_w12_tfall"
  $done = Join-Path $outdir "directed_search_frontier_$rid.csv"
  $ckpt = Join-Path $outdir "checkpoint_frontier_$rid.csv"
  if (Test-Path $done) { Log "SKIP  $sc $tg (finished)"; return }
  $env:GRID_STEP_MULT = $mult; $env:QUEUE_ITEM = "$i"
  $a = @("directed_search.py","--scenario",$sc,"--target",$tg,
         "--climate-year","2009","--demand-2040","--h2-2040","--ev-2040",
         "--p2g-endogenous","--n-weeks","12","--n-restarts","4",
         "--max-evals","1500","--n-workers","4","--no-monotone","--no-validate")
  if (Test-Path $ckpt) { $a += "--resume" }
  Log "START $sc $tg (resume=$(Test-Path $ckpt), grid x$mult)"
  & $py queue_progress.py item_start $Queue $i
  $t0 = Get-Date
  & $py @a 2>&1 | Tee-Object -FilePath (Join-Path $logdir "rerun_$($sc)_$($tg).log") -Append
  $h = [math]::Round(((Get-Date) - $t0).TotalHours, 1)
  if (Test-Path $done) { Log "DONE  $sc $tg ($h h)"; & $py queue_progress.py item_done $Queue $i $h }
  else { Log "WARN  $sc $tg ($h h, no final file - rerun this script to resume)"; & $py notify.py "WARNING queue $Queue : $sc $tg stopped without final file after $h h - rerun the script to resume" }
  $env:GRID_STEP_MULT = ""
}
function Run-Grid($i, $sc) {
  $out  = "output\grid_floor_5node_$($sc)_tfall.csv"
  $mark = "$out.done"
  if (Test-Path $mark) { Log "SKIP  grid $sc (finished)"; return }
  $env:QUEUE_ITEM = "$i"
  Log "START grid $sc (resumes from existing rows if any)"
  & $py queue_progress.py item_start $Queue $i
  $t0 = Get-Date
  & $py grid_floor.py --scenario $sc --n-weeks 12 --climate-year 2009 --demand-2040 `
    --h2-2040 --ev-2040 --p2g-endogenous --fracs-sc 0 --fracs-dsmev "0,0.5,1.0" `
    --out $out 2>&1 | Tee-Object (Join-Path $logdir "rerun_grid_$sc.log") -Append
  $h = [math]::Round(((Get-Date) - $t0).TotalHours, 1)
  if ($LASTEXITCODE -eq 0) { New-Item -ItemType File -Force -Path $mark | Out-Null; Log "DONE  grid $sc ($h h)"; & $py queue_progress.py item_done $Queue $i $h }
  else { Log "WARN  grid $sc exit $LASTEXITCODE"; & $py notify.py "WARNING queue $Queue : grid $sc exit $LASTEXITCODE - rerun the script to resume" }
}

Log "=== run_rerun_phase1.ps1 START (TYNDP_FIXES=all, LP threads $thr) ==="
& $py queue_progress.py start $Queue
for ($i = 0; $i -lt $items.Count; $i++) {
  $it = $items[$i]
  if ($it.k -eq "f") { Run-Frontier $i $it.sc $it.tg $it.m } else { Run-Grid $i $it.sc }
}
& $py queue_progress.py finished $Queue
$env:TYNDP_FIXES = ""; $env:QUEUE_NAME = ""; $env:QUEUE_ITEM = ""
Log "=== run_rerun_phase1.ps1 FINISHED ==="
