# Single-fibre simulations

This directory contains scripts for generating, thermalising and heating individual W fibres with internal cavities and He. The simulations are organised through configuration summaries generated from a YAML file.

## 1. Structure generation

From the repository root:

```bash
cd single_fiber/prepare
python generate.py
```

`series.yml` specifies the potential, fibre geometry, porosity, bubble-size distribution, He/V ratios and random seeds. The generator writes LAMMPS data files to `structures/<potential>/<case>/s<seed>/` and creates `summaries/<case>.csv` for each case defined in the YAML file. Each summary row identifies one geometry and He-filling state, including its thermalisation seed and input structure path. The summary rows are selected using Slurm array indices.

## 2. Thermalisation

Submit jobs from `single_fiber/prepare/`:

```bash
sbatch --array=<indices> thermalise.sh <case>
```

`<case>` is the summary filename without `.csv`. The array selects entries from `summaries/<case>.csv`, with `SLURM_ARRAY_TASK_ID=0` corresponding to the first data row after the header. Each task reads the specified structure and writes a `300K.restart` in `prepared/<potential>/<case>/s<seed>/<state>/`. Use `0-(N-1)` to select all `N` summary rows or a Slurm index list/range to select a subset. Complete the required thermalisation jobs before starting heating.

## 3. Heating

From `single_fiber/heat/`:

```bash
sbatch --array=<indices> heat.sh <case> <alpha>
```

`<alpha>` is the nominal heating rate in K/ps. Each task uses the corresponding row of `prepare/summaries/<case>.csv` to locate its thermalised restart. Results are stored in `heat/results/<potential>/<case>/s<seed>/<state>/alpha_<rate>/`.
