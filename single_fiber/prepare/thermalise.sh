#!/bin/bash
#SBATCH --job-name=fiber-therm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=therm-%A_%a.out

set -eo pipefail
export OMP_TOOL=disabled
unset OMP_TOOL_LIBRARIES
export TSAN_OPTIONS=ignore_noninstrumented_modules=1
export OMP_NUM_THREADS=1

source /opt/intel/oneapi/setvars.sh

PREPARE="$SLURM_SUBMIT_DIR"
LMP="$HOME/lammps-10Sep2025_cpu/build_shared/lmp"
POTFILE="$HOME/potentials/WHHe_TMB_JPCM_2025.eam.he"
row=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PREPARE/summaries/$1.csv")
test -n "$row" || { echo "Missing summary row: $1 task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
IFS=, read -r POT CASE STATE GSEED TSEED DATA_REL REST <<< "$row"
test "$POT" = TMB
test -s "$PREPARE/$DATA_REL"
test -s "$POTFILE"

OUT="$PREPARE/prepared/TMB/$CASE/s$GSEED/$STATE"
TMP="$PREPARE/../tmp/thermalise_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [ -s "$OUT/300K.restart" ] && [ -s "$OUT/data.300K" ] && [ -s "$OUT/log.lammps" ]; then
    echo "Prepared state exists: $OUT"
    exit 0
fi
test ! -e "$OUT/300K.restart" && test ! -e "$OUT/data.300K" && test ! -e "$OUT/log.lammps"
test ! -e "$TMP"

mkdir -p "$TMP" "$OUT"
cp "$PREPARE/$DATA_REL" "$TMP/input.data"
cp "$PREPARE/thermalise.in" "$POTFILE" "$TMP/"

cd "$TMP"
mpirun -np "$SLURM_NTASKS" "$LMP" \
    -log log.lammps \
    -var data_in input.data \
    -var thermal_seed "$TSEED" \
    -var restart_out 300K.restart \
    -var data_out data.300K \
    -in thermalise.in

cp 300K.restart data.300K log.lammps "$OUT/"
rm -rf "$TMP"
