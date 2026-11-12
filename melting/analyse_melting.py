import csv
from pathlib import Path

import numpy as np


root = Path(__file__).resolve().parent
results = root / "results_nph" / "T_5100"

dt_ps = 0.001
start_ps = 200.0
end_ps = 500.0
fmin = 0.25
fmax = 0.75
t95 = 2.262


def load_nph(path):
    data = np.loadtxt(str(path), comments="#")
    time = data[:, 0] * dt_ps
    mask = (time >= start_ps) & (time <= end_ps)

    if time[-1] < end_ps - 0.2:
        raise ValueError("Incomplete NPH trajectory")

    return data[mask]


def load_profiles(path):
    frames = []

    with path.open() as f:
        while True:
            line = f.readline()

            if not line:
                break

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) != 3:
                continue

            step = int(float(fields[0]))
            nchunk = int(float(fields[1]))
            bins = []

            for _ in range(nchunk):
                row = f.readline().split()
                bins.append([float(value) for value in row[:5]])

            frames.append((step * dt_ps, np.asarray(bins)))

    return frames


def calibrate(frames, column):
    solid_sum = 0.0
    solid_count = 0.0
    liquid_sum = 0.0
    liquid_count = 0.0

    for _, bins in frames[:3]:
        z = bins[:, 1]
        count = bins[:, 2]
        value = bins[:, column]

        solid = (z >= 0.10) & (z < 0.40)
        liquid = (z >= 0.60) & (z < 0.90)

        solid_sum += np.sum(count[solid] * value[solid])
        solid_count += np.sum(count[solid])
        liquid_sum += np.sum(count[liquid] * value[liquid])
        liquid_count += np.sum(count[liquid])

    solid_value = solid_sum / solid_count
    liquid_value = liquid_sum / liquid_count

    return solid_value, liquid_value, 0.5 * (solid_value + liquid_value)


def phase_fraction(frames, column, solid_value, liquid_value, threshold):
    fractions = []
    solid_is_lower = solid_value < liquid_value

    for time, bins in frames:
        if time < start_ps or time > end_ps:
            continue

        count = bins[:, 2]
        value = bins[:, column]
        solid = value < threshold if solid_is_lower else value > threshold
        fractions.append(np.sum(count[solid]) / np.sum(count))

    return float(np.mean(fractions))


def analyze(rep):
    directory = results / f"rep_{rep}"
    nph = load_nph(directory / "nph.txt")
    profiles = load_profiles(directory / "phase_profile.txt")

    csp_solid, csp_liquid, csp_threshold = calibrate(profiles, 3)
    pe_solid, pe_liquid, pe_threshold = calibrate(profiles, 4)

    f_csp = phase_fraction(profiles, 3, csp_solid, csp_liquid, csp_threshold)
    f_pe = phase_fraction(profiles, 4, pe_solid, pe_liquid, pe_threshold)

    return {
        "rep": rep,
        "Tmean_K": float(np.mean(nph[:, 1])),
        "Pmean_bar": float(np.mean(nph[:, 2])),
        "f_csp": f_csp,
        "f_pe": f_pe,
        "valid": fmin <= f_csp <= fmax and fmin <= f_pe <= fmax,
    }


rows = [analyze(rep) for rep in range(10)]

with (root / "Tm_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        f"{row['rep']:2d}  T={row['Tmean_K']:.2f} K  P={row['Pmean_bar']:.1f} bar  fCSP={row['f_csp']:.3f}  fPE={row['f_pe']:.3f}  valid={row['valid']}"
    )

if not all(row["valid"] for row in rows):
    raise RuntimeError("One or more replicas do not retain solid-liquid coexistence")

temperatures = np.array([row["Tmean_K"] for row in rows])
tm = np.mean(temperatures)
sd = np.std(temperatures, ddof=1)
sem = sd / np.sqrt(len(temperatures))
ci95 = t95 * sem

print()
print(f"Tm = {tm:.2f} K")
print(f"SD = {sd:.2f} K")
print(f"SEM = {sem:.2f} K")
print(f"95% CI = {tm:.2f} ± {ci95:.2f} K")
