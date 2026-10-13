# The cost frontier of renewable cannibalization

Code and data for the paper "Mitigating, not eliminating: the cost frontier of
renewable cannibalization in interconnected power markets" by Zdeněk
Pustějovský, Prague University of Economics and Business
(zdenek.pustejovsky@vse.cz). Manuscript under preparation.

The model traces least-cost frontiers of the renewable value factor. For
every target level it finds the cheapest portfolio of battery storage,
demand-side management and interconnection that reaches the target in a
five-node European power system (DE, CZ, DK, AT, NL) with endogenous prices.
The system is calibrated to the ENTSO-E/ENTSOG TYNDP 2024 scenarios
(National Trends 2030, Distributed Energy 2035, 2040, 2050). An evolutionary
outer search (CMA-ES) over the flexibility portfolio calls a linear
investment and dispatch model (Pyomo) on 12 representative weeks.

## Repository layout

```
time_series.xlsx            DIETER hourly input sheet, five modelled countries
model/
  data_pipeline.py          representative weeks, PECD capacity factors and demand
  scenarios.py              scenario definitions and application to the data
  scenario_data.py          TYNDP 2024 capacities, demand, EV fleet per scenario
  tyndp_fixes.py            TYNDP system components (offshore hubs, hydro, other
                            generation, hydrogen CCGTs, gas fleet, fuel prices,
                            demand response, oil, battery potentials, hourly
                            border prices), switched on with TYNDP_FIXES=all
  model_builder.py          linear investment and dispatch model
  directed_search.py        CMA-ES frontier search, restarts, polish, first-order check
  grid_floor.py             factorial grid over uniformly scaled portfolios
  node_factorial.py         node-level interaction experiments
  tyndp_battery_benchmark.py  cost optimum with the TYNDP battery fleet
  queue_progress.py, notify.py  optional Telegram progress messages
  run_final_baselines.ps1   baselines and battery benchmark
  run_rerun_phase1.ps1      all frontiers and factorial grids (two parallel queues)
  run_rerun_phase2.ps1      ceiling probes, robustness variants, interaction experiments
  phase2_plan.py            rules that derive the phase-2 experiments from the frontiers
  run_phase2.py             phase-2 runner (waits for phase 1, then runs the plan)
  tyndp 2040/
    *.json                  TYNDP 2024 inputs extracted for the five nodes
    extract_all.py          recreates every JSON input from the raw TYNDP files
    extract_*.py            the individual extraction steps
    PECD/                   PECD 2021.3 subset, five countries, all climate years
paper/
  make_paper_numbers.py     every number, table and figure of the paper
  build_numbers.py          section-by-section driver with placeholders
```

## Installation

Python 3.11 or newer.

```
pip install -r requirements.txt
```

All reported runs used Gurobi (academic licence) through Pyomo. HiGHS is a
free fallback, slower on the 12-week model.

## Running

All commands from `model/`. Outputs go to `model/output/` and `model/logs/`.
Every run resumes from its last finished target when restarted with the same
command.

Baselines and the TYNDP battery benchmark (about 30 minutes):

```
.\run_final_baselines.ps1
```

All frontiers and factorial grids (several days on 8 cores, two windows):

```
.\run_rerun_phase1.ps1 -Queue A
.\run_rerun_phase1.ps1 -Queue B
```

Phase 2 (ceiling probes, robustness variants, interaction experiments, about
three days). Start it next to phase 1, it waits until phase 1 has finished:

```
.\run_rerun_phase2.ps1 -Queue A
.\run_rerun_phase2.ps1 -Queue B
```

Phase 2 derives its experiments from the phase-1 frontiers by fixed rules
(`phase2_plan.py`): the elbow of a frontier by the Kneedle rule (Satopää et
al., 2011), the weak node as the node of the binding value factor at the
ceiling, probe targets 0.02 and 0.06 above the ceiling, and sweep values as
fixed fractions of the deployment potentials. The plan is written to
`model/logs/phase2_plan.json`. `.\run_rerun_phase2.ps1 -Queue B -DryRun`
prints the commands without running them.

A single frontier by hand, for example DE 2040 with the weakest-link target:

```
set TYNDP_FIXES=all
python directed_search.py --scenario de2040 --target vf --climate-year 2009 --demand-2040 --h2-2040 --ev-2040 --p2g-endogenous --n-weeks 12 --n-restarts 4 --max-evals 1500 --n-workers 4 --no-monotone --no-validate
```

Use `--target vfw` for the generation-weighted (fleet-average) value factor
and `--baseline-only` for the cost-optimal system alone.

Paper numbers, tables and figures:

```
cd paper
set PAPER_SET=tfall
python build_numbers.py
```

Telegram messages are sent only if the environment variables
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set.

## Data

All data needed to run the model is in this repository. The files are
third-party data, redistributed under their licences, with the changes listed.

| Data | Source and licence | Changes |
|---|---|---|
| `model/tyndp 2040/PECD/*.parquet` | ENTSO-E Pan-European Climatic Database 2021.3, Parquet version by M. De Felice (2021), Zenodo, [doi:10.5281/zenodo.5780185](https://doi.org/10.5281/zenodo.5780185), CC BY 4.0 | rows limited to DE, CZ, DK, AT, NL, National Estimates 2030 files only |
| `model/tyndp 2040/*.json` | TYNDP 2024 Scenarios, ENTSO-E and ENTSOG, [download page](https://2024.entsos-tyndp-scenarios.eu/download/), CC BY 4.0 | values extracted for the five nodes and four scenarios |
| `time_series.xlsx` | DIETER model input data (Zerrahn and Schill), compiled from TYNDP, Open Power System Data and renewables.ninja (the renewables.ninja series are CC BY-NC 4.0) | sheet `basic` only, five countries only |

Attribution: "TYNDP 2024 Scenarios", ENTSO-E and ENTSOG.

In the reported configuration (`--climate-year 2009`) the solar, wind and
demand series of `time_series.xlsx` are replaced by PECD data. Only its
run-of-river and reservoir inflow series enter the results.

### Raw TYNDP files

The raw TYNDP 2024 files are needed only to re-run the extraction scripts.
They are about 400 MB, which is too large for this repository. They will be
archived with the Zenodo release of the paper. Until then, download them from
the [TYNDP 2024 Scenarios download page](https://2024.entsos-tyndp-scenarios.eu/download/)
and unpack them into `model/tyndp 2040/`:

| File(s) | Download |
|---|---|
| `MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx` | [NT+ 2030, climate year 2009](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx.zip) |
| `DE2035CY2009/` | [Distributed Energy 2035, climate year 2009](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/DE2035CY2009.zip) |
| `MMStandardOutputFile_DE2040_Plexos_CY2009_*.xlsb` (directly in `tyndp 2040/`) | [Distributed Energy 2040, climate year 2009](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/DE2040CY2009.zip) |
| `DE2050CY2009/` | [Distributed Energy 2050, climate year 2009](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/DE2050CY2009.zip) |
| `EV Modelling Inputs/` | [EV Modelling Inputs](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-inputs/EV-Modelling-Inputs.zip) |
| commodity prices | [Commodity Prices](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-inputs/Prices.zip) |

Then (about 20 minutes):

```
cd "model/tyndp 2040"
python extract_all.py --check     # rebuild into _check/ and compare with the JSON files
python extract_all.py             # or overwrite the JSON files
```

## Licence

Code: MIT (see `LICENSE`). Data: see the table above.

## Citation

Pustějovský, Z. (2026). Mitigating, not eliminating: the cost frontier of
renewable cannibalization in interconnected power markets. Manuscript.
