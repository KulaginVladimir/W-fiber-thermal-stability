from pathlib import Path

import yaml


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_lammps_data(path, structure, padding):
    xlo = -structure.radius - padding
    xhi = structure.radius + padding
    ylo = -structure.radius - padding
    yhi = structure.radius + padding

    with Path(path).open("w", encoding="utf-8") as f:
        f.write("W nanofiber\n\n")
        f.write(f"{len(structure.w) + len(structure.he)} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{xlo:.10f} {xhi:.10f} xlo xhi\n")
        f.write(f"{ylo:.10f} {yhi:.10f} ylo yhi\n")
        f.write(f"0.0000000000 {structure.length:.10f} zlo zhi\n\n")
        f.write("Atoms # atomic\n\n")

        atom_id = 1
        for x, y, z in structure.w:
            f.write(f"{atom_id} 1 {x:.10f} {y:.10f} {z:.10f}\n")
            atom_id += 1
        for x, y, z in structure.he:
            f.write(f"{atom_id} 2 {x:.10f} {y:.10f} {z:.10f}\n")
            atom_id += 1
