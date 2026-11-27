#!/bin/bash
#SBATCH --job-name=pair-heat
#SBATCH --partition=normal
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=pair-heat-%A_%a.out

set -e

export OMP_TOOL=disabled
unset OMP_TOOL_LIBRARIES
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export OMP_NUM_THREADS=1

source /opt/intel/oneapi/setvars.sh

HEAT="$SLURM_SUBMIT_DIR"
TWO="$(cd "$HEAT/.." && pwd)"
PREPARE="$TWO/prepare"
LMP="$HOME/lammps-10Sep2025_cpu/build_shared/lmp"

export PYTHONPATH="$HOME/lammps-10Sep2025_cpu/python:$HOME/.local/lib/python3.6/site-packages"
export LD_LIBRARY_PATH="$HOME/lammps-10Sep2025_cpu/build_shared:${LD_LIBRARY_PATH:-}"

ALPHA=$(printf "%.1f" "$1")
SUMMARY="$PREPARE/structures/summary.csv"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$SUMMARY")
IFS=',' read -r CASE GAP SEP TSEED DATA_REL NSOURCE NTARGET <<< "$row"

RESTART="$PREPARE/prepared/TMB/$CASE/300K.restart"
OUT="$HEAT/results/TMB/$CASE/alpha_${ALPHA}"
TMP="$TWO/tmp/heating_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TMP" "$OUT"

cp "$RESTART" "$TMP/input.restart"
cp "$HEAT/heat_pair.in" "$TMP/heat.in"
cp "$HEAT/connectivity_pair.py" "$TMP/"
cp "$HEAT/ejecta_pair.py" "$TMP/"
cp "$HOME/potentials/WHHe_TMB_JPCM_2025.eam.he" "$TMP/"

cd "$TMP"

mpirun -np "$SLURM_NTASKS" "$LMP" \
    -log log.lammps \
    -var restart_in input.restart \
    -var alpha "$ALPHA" \
    -var Tmax 20000.0 \
    -in heat.in

gzip -f w_clusters.txt
cp initial_state.txt heat.txt impact.txt heat_off.txt final.data log.lammps w_clusters.txt.gz he_release.txt "$OUT/"

if [ -f dump_mech.lammpstrj.gz ]; then
    cp dump_mech.lammpstrj.gz "$OUT/"
fi

rm -rf "$TMP"
