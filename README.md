# Stability of tungsten nanofibres under rapid heating

## Overview

This repository contains LAMMPS scripts and processed molecular dynamics (MD) data used to study the rapid heating and destruction of tungsten nanofibres containing helium bubbles.

The calculations reported in the manuscript use the W–H–He potential by Tirumala et al. (*Journal of Physics: Condensed Matter* **38** (2026) 075703. [DOI: 10.1088/1361-648X/ae37bc](https://doi.org/10.1088/1361-648X/ae37bc)). Full MD trajectories are excluded from the repository. The processed data needed to reproduce the manuscript figures are provided.

## Repository structure

```text
single_fiber/           Preparation and heating of individual fibres
  prepare/              Geometry generation and thermalisation
  heat/                 Heating and ejecta diagnostics
two_fiber/              Preparation and heating of fibre pairs
  prepare/              Geometry generation and thermalisation
  heat/                 Heating and debris impact analysis
melting/                NPH solid–liquid coexistence calculations and results
postprocessing/         Analysis notebooks
  pp_data/              Processed data for the figures
  figures/              Figures and representative OVITO snapshots
```

## Python environment

Use Python 3.11 or newer. On Windows (PowerShell), create the environment in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To reproduce the figures, start Jupyter from `postprocessing/`:

```powershell
cd postprocessing
jupyter lab
```

Python dependencies for MD calculations are listed in `requirements-md.txt`. The LAMMPS Python module must correspond to the installed LAMMPS build and use the same MPI implementation as mpi4py.

## MD calculations

The simulations were performed with [LAMMPS (10 Sep 2025)](https://github.com/lammps/lammps/releases/tag/patch_10Sep2025) using `pair_style eam/he`. Download `WHHe_TMB_JPCM_2025.eam.he` from the [NIST Interatomic Potentials Repository](https://www.ctcms.nist.gov/potentials/entry/2026--Tirumala-A-S-Shattock-O-Mason-D-R-et-al--W-H-He/) and place it at the location expected by the Slurm scripts (or update the local potential path).

The `melting/` directory includes ten NPH solid–liquid coexistence runs and their phase diagnostics. Recalculate the reported melting-temperature estimate with:

```powershell
python melting/analyse_melting.py
```
