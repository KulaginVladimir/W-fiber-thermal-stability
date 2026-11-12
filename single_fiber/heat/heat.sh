#!/bin/bash
#SBATCH --job-name=fiber-heat
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=heat-%A_%a.out

set -eo pipefail

export OMP_TOOL=disabled
unset OMP_TOOL_LIBRARIES
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export OMP_NUM_THREADS=1

source /opt/intel/oneapi/setvars.sh

HEAT="$SLURM_SUBMIT_DIR"
SINGLE_FIBER="$(cd "$HEAT/.." && pwd)"
PREPARE="$SINGLE_FIBER/prepare"
LMP="$HOME/lammps-10Sep2025_cpu/build_shared/lmp"

export PYTHONPATH="$HOME/lammps-10Sep2025_cpu/python:$HOME/.local/lib/python3.6/site-packages"
export LD_LIBRARY_PATH="$HOME/lammps-10Sep2025_cpu/build_shared:${LD_LIBRARY_PATH:-}"

CASE_RUN="$1"
ALPHA=$(printf "%.1f" "$2")
TMAX=20000.0

SUMMARY="$PREPARE/summaries/$CASE_RUN.csv"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$SUMMARY")
IFS=',' read -r POT CASE STATE GSEED TSEED DATA_REL REST <<< "$row"

RESTART="$PREPARE/prepared/TMB/$CASE/s$GSEED/$STATE/300K.restart"
OUT="$HEAT/results/TMB/$CASE/s$GSEED/$STATE/alpha_${ALPHA}"
TMP="$SINGLE_FIBER/tmp/heating_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
POTFILE="$HOME/potentials/WHHe_TMB_JPCM_2025.eam.he"

mkdir -p "$TMP" "$OUT"
cp "$RESTART" "$TMP/input.restart"
cp "$HEAT/heat.in" "$HEAT/connectivity.py" "$HEAT/ejecta.py" "$TMP/"
cp "$POTFILE" "$TMP/"

cd "$TMP"

mpirun -np "$SLURM_NTASKS" "$LMP" \
    -log log.lammps \
    -var restart_in input.restart \
    -var alpha "$ALPHA" \
    -var Tmax "$TMAX" \
    -var out_dir . \
    -in heat.in

gzip -f w_clusters.txt w_cluster_energy.txt

cp heat.txt structure.txt heat_off.txt \
   he_release.txt w_clusters.txt.gz w_cluster_energy.txt.gz final.data log.lammps "$OUT/"

if [ -s dump_mech.lammpstrj ]; then
    gzip -f dump_mech.lammpstrj
    cp dump_mech.lammpstrj.gz "$OUT/"
fi

rm -rf "$TMP"