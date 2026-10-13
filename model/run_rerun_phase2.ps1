# ============================================================================
# run_rerun_phase2.ps1 -- phase 2 of the rerun (TYNDP_FIXES=all): ceiling
# probes, robustness variants, interaction experiments. Thin wrapper around
# run_phase2.py (all logic and the plan rules are there and in phase2_plan.py).
# USAGE: start now, next to the running phase-1 windows:
#   window 3:  .\run_rerun_phase2.ps1 -Queue A
#   window 4:  .\run_rerun_phase2.ps1 -Queue B
# Each waits (no CPU) until its phase-1 queue has finished, then runs.
# CRASH / RESTART: run the same command again; finished items are skipped.
# Preview the commands without running anything:
#   .\run_rerun_phase2.ps1 -Queue B -DryRun
# ============================================================================
param([Parameter(Mandatory=$true)][ValidateSet("A","B")][string]$Queue, [switch]$DryRun)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$py = "python"
if (-not (Test-Path $py)) { $py = "python" }
$a = @("run_phase2.py", "--queue", $Queue)
if ($DryRun) { $a += "--dry-run" }
& $py @a
