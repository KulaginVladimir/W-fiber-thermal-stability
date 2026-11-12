#!/bin/bash
#SBATCH --job-name=TMB-Tm-NPH10
#SBATCH --partition=normal
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=Tm-NPH10-%A_%a.out

export OMP_TOOL=disabled
unset OMP_TOOL_LIBRARIES
export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
export OMP_NUM_THREADS=1

source /opt/intel/oneapi/setvars.sh
set -e

MELTING="$SLURM_SUBMIT_DIR"
ROOT="$(cd "$MELTING/.." && pwd)"

LMP="$HOME/lammps-10Sep2025_cpu/build_shared/lmp"
POT="$HOME/potentials/WHHe_TMB_JPCM_2025.eam.he"
INPUT="melting_coexistence_nph.in"

export LD_LIBRARY_PATH="$HOME/lammps-10Sep2025_cpu/build_shared:${LD_LIBRARY_PATH:-}"

TINIT=5100
REP="$SLURM_ARRAY_TASK_ID"
SEED=$((51027 + 10000 * REP))

OUT="$MELTING/results_nph/T_${TINIT}/rep_${REP}"
WORK="$ROOT/tmp/melting_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [ -e "$OUT" ]; then
    echo "Output exists: $OUT" >&2
    exit 1
fi

mkdir -p "$OUT" "$WORK"

cp "$MELTING/$INPUT" "$WORK/"
cp "$POT" "$WORK/WHHe_TMB_JPCM_2025.eam.he"

cd "$WORK"

mpirun -np "$SLURM_NTASKS" "$LMP" -log log.lammps -var Tinit "$TINIT" -var seed "$SEED" -var potfile WHHe_TMB_JPCM_2025.eam.he -var out_dir . -in "$INPUT"

cp nph.txt phase_profile.txt initial_coexistence.data final.data log.lammps "$OUT/"

rm -rf "$WORK"