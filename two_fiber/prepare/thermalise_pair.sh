#!/bin/bash
#SBATCH --job-name=pair-therm
#SBATCH --partition=normal
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=therm-pair-%A_%a.out

set -e
export OMP_TOOL=disabled
unset OMP_TOOL_LIBRARIES
export TSAN_OPTIONS=ignore_noninstrumented_modules=1
export OMP_NUM_THREADS=1

source /opt/intel/oneapi/setvars.sh

PREPARE="$SLURM_SUBMIT_DIR"
TWO="$(cd "$PREPARE/.." && pwd)"
LMP="$HOME/lammps-10Sep2025_cpu/build_shared/lmp"
IMPI=/opt/intel/oneapi/mpi/2021.15
export LD_LIBRARY_PATH="$HOME/lammps-10Sep2025_cpu/build_shared:${LD_LIBRARY_PATH:-}"

SUMMARY="$PREPARE/structures/summary.csv"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$SUMMARY" | tr -d '\r')
IFS=',' read -r CASE GAP SEP TSEED DATA_REL NSOURCE NTARGET <<< "$row"

DATA_IN="$PREPARE/$DATA_REL"
OUT="$PREPARE/prepared/TMB/$CASE"
TMP="$TWO/tmp/thermalise_${CASE}_${SLURM_JOB_ID}"

mkdir -p "$OUT" "$TMP"
cp "$DATA_IN" "$TMP/input.data"
cp "$PREPARE/thermalise_pair.in" "$TMP/thermalise_pair.in"
cp "$HOME/potentials/WHHe_TMB_JPCM_2025.eam.he" "$TMP/"

BASE=$((TSEED + 300))

cd "$TMP"
"$IMPI/bin/mpiexec.hydra" -n "$SLURM_NTASKS" "$LMP" \
    -log log.lammps \
    -var DATA input.data \
    -var SEED_INIT "$((BASE + 1))" \
    -var SEED_RAMP_SOURCE "$((BASE + 2))" \
    -var SEED_RAMP_TARGET "$((BASE + 3))" \
    -var SEED_HOLD_SOURCE "$((BASE + 4))" \
    -var SEED_HOLD_TARGET "$((BASE + 5))" \
    -in thermalise_pair.in

cp 300K.restart data.300K log.lammps "$OUT/"
rm -rf "$TMP"
