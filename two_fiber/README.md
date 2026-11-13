# Two-fibre simulations

This directory contains scripts for generating, thermalising and heating pairs of parallel W fibres. Energy is supplied to the source fibre, while the target interacts with the emitted products. The simulations are organised through a configuration summary generated from a YAML file.

## 1. Structure generation

Generate the individual fibre structures first, as described in `single_fiber/README.md`. Then, from the repository root:

```bash
cd two_fiber/prepare
python generate_pairs.py series_two_fiber.yml
```

`series_two_fiber.yml` specifies the source and target structures, geometry pairs, separation distances and output paths. The generator writes LAMMPS data files to `structures/<potential>/<case>/` and creates `structures/summary.csv`. Each summary row identifies one pair and separation, including its thermalisation seed and input structure path. The summary rows are selected using Slurm array indices.

## 2. Thermalisation

Submit jobs from `two_fiber/prepare/`:

```bash
sbatch --array=<indices> thermalise_pair.sh
```

The array selects entries from `structures/summary.csv`, with `SLURM_ARRAY_TASK_ID=0` corresponding to the first data row after the header. Each task reads the specified pair and writes a `300K.restart` in `prepared/<potential>/<case>/`. Use `0-(N-1)` to select all `N` summary rows or a Slurm index list/range to select a subset. Complete the required thermalisation jobs before starting heating.

## 3. Heating

From `two_fiber/heat/`:

```bash
sbatch --array=<indices> heat_pair.sh <alpha>
```

`<alpha>` is the nominal heating rate in K/ps. Each task uses the corresponding row of `prepare/structures/summary.csv` to locate its thermalised restart. Results are stored in `heat/results/<potential>/<case>/alpha_<rate>/`.
