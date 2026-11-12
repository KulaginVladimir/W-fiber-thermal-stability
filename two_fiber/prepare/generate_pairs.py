from pathlib import Path
import csv
import sys
import yaml


def read_data(path):
    bounds, atoms = {}, []
    in_atoms = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        p = s.split()
        if len(p) >= 4 and p[-2:] == ["xlo", "xhi"]:
            bounds["x"] = (float(p[0]), float(p[1]))
        elif len(p) >= 4 and p[-2:] == ["ylo", "yhi"]:
            bounds["y"] = (float(p[0]), float(p[1]))
        elif len(p) >= 4 and p[-2:] == ["zlo", "zhi"]:
            bounds["z"] = (float(p[0]), float(p[1]))
        elif s.startswith("Atoms"):
            in_atoms = True
        elif in_atoms:
            if s[0].isalpha():
                break
            if len(p) >= 5 and s[0].isdigit():
                atoms.append((int(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return bounds, atoms


def write_pair(path, source, target, sb, tb, radius, gap, padding):
    separation = 2 * radius + gap
    xhalf = separation / 2 + radius + padding
    yhalf = radius + padding
    sx, sy = (sum(sb[k]) / 2 for k in ("x", "y"))
    tx, ty = (sum(tb[k]) / 2 for k in ("x", "y"))
    atoms = []
    for typ, x, y, z in source:
        atoms.append((len(atoms) + 1, typ, x - sx - separation / 2, y - sy, z))
    for typ, x, y, z in target:
        atoms.append((len(atoms) + 1, typ + 2, x - tx + separation / 2, y - ty, z))

    with path.open("w") as f:
        f.write("Two-fiber W-He structure\n\n")
        f.write(f"{len(atoms)} atoms\n4 atom types\n\n")
        f.write(f"{-xhalf:.10f} {xhalf:.10f} xlo xhi\n")
        f.write(f"{-yhalf:.10f} {yhalf:.10f} ylo yhi\n")
        f.write(f"{sb['z'][0]:.10f} {sb['z'][1]:.10f} zlo zhi\n\n")
        f.write(
            "Masses\n\n1 183.84\n2 4.002602\n3 183.84\n4 4.002602\n\nAtoms # atomic\n\n"
        )
        for i, typ, x, y, z in atoms:
            f.write(f"{i} {typ} {x:.10f} {y:.10f} {z:.10f}\n")


config = Path(sys.argv[1]).resolve()
base = config.parent
cfg = yaml.safe_load(config.read_text())
radius = cfg["radius_lattice"] * cfg["lattice_parameter_angstrom"]
rows = []

for source_seed, target_seed in cfg["pairs"]:
    source_path = base / cfg["input_template"].format(**cfg, seed=source_seed)
    target_path = base / cfg["input_template"].format(**cfg, seed=target_seed)
    sb, source = read_data(source_path)
    tb, target = read_data(target_path)

    for item in cfg["gaps"]:
        gap = item["gap_angstrom"]
        case = f"gap{int(gap)}_s{source_seed}s{target_seed}"
        out = base / cfg["output_template"].format(**cfg, case=case)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_pair(
            out, source, target, sb, tb, radius, gap, cfg["box_padding_angstrom"]
        )
        rows.append(
            (
                case,
                gap,
                2 * radius + gap,
                item["thermal_seed_base"] + source_seed,
                str(out.relative_to(base)),
                len(source),
                len(target),
            )
        )

summary = base / cfg["summary_file"]
summary.parent.mkdir(parents=True, exist_ok=True)
with summary.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("CASE", "GAP", "SEP", "TSEED", "DATA_REL", "NSOURCE", "NTARGET"))
    writer.writerows(rows)
