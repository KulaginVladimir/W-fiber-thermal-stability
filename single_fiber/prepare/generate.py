import csv
import sys
from copy import deepcopy
from pathlib import Path

from geometry import generate_structures, statistics
from structure_io import load_yaml, write_lammps_data, write_yaml

root = Path(__file__).resolve().parent

fields = [
    "potential",
    "case",
    "state",
    "geometry_seed",
    "thermal_seed",
    "data_file",
    "w_initial",
    "w_final",
    "w_removed",
    "void_fraction",
    "he_atoms",
    "he_per_vacancy",
    "bubble_count",
    "bubble_d_mean_A",
    "bubble_d_std_A",
    "bubble_d_min_A",
    "bubble_d_max_A",
]


def make_row(
    potential, case_name, state, geometry_seed, thermal_seed, data_path, stats
):
    diameter = stats.get("bubble_diameter_angstrom") or {}
    return {
        "potential": potential,
        "case": case_name,
        "state": state,
        "geometry_seed": geometry_seed,
        "thermal_seed": thermal_seed,
        "data_file": str(data_path.relative_to(root)),
        "w_initial": stats.get("w_initial", ""),
        "w_final": stats.get("w_final", ""),
        "w_removed": stats.get("w_removed", ""),
        "void_fraction": stats.get("void_fraction", ""),
        "he_atoms": stats.get("he_atoms", ""),
        "he_per_vacancy": stats.get("he_per_vacancy", ""),
        "bubble_count": stats.get("bubble_count", ""),
        "bubble_d_mean_A": diameter.get("mean", ""),
        "bubble_d_std_A": diameter.get("std", ""),
        "bubble_d_min_A": diameter.get("min", ""),
        "bubble_d_max_A": diameter.get("max", ""),
    }


def main():
    config = root / (sys.argv[1] if len(sys.argv) > 1 else "series.yml")
    cfg = load_yaml(config)

    potential = "TMB"
    if cfg["potential"]["name"] != potential:
        raise ValueError("Expected TMB in series.yml")
    lattice_parameter = float(cfg["potential"]["lattice_parameter_angstrom"])
    thermal_seed_base = int(cfg["thermal_seed_base"])

    (root / "summaries").mkdir(exist_ok=True)

    for case_cfg in cfg["cases"]:
        case_name = case_cfg["name"]
        summary = root / "summaries" / f"{case_name}.csv"

        fiber = deepcopy(cfg["fiber"])
        fiber.update(case_cfg.get("fiber", {}))
        fiber["lattice_parameter_angstrom"] = lattice_parameter
        padding = float(fiber["box_padding_xy_angstrom"])

        rows = []

        for geometry_seed in case_cfg["geometry_seeds"]:
            geometry_seed = int(geometry_seed)
            thermal_seed = thermal_seed_base + geometry_seed

            for fill in case_cfg["fillings"]:
                fill = float(fill)
                state = f"he{round(fill)}"

                structure_dir = (
                    root / "structures" / potential / case_name / f"s{geometry_seed}"
                )
                data_path = structure_dir / f"data.{state}"
                yml_path = structure_dir / f"{state}.yml"

                if data_path.is_file() and yml_path.is_file():
                    meta = load_yaml(yml_path)
                    rows.append(
                        make_row(
                            potential,
                            case_name,
                            state,
                            geometry_seed,
                            thermal_seed,
                            data_path,
                            meta.get("statistics", {}),
                        )
                    )
                    continue

                helium = deepcopy(cfg["helium"])
                helium["he_per_vacancy"] = fill

                run_cfg = {
                    "fiber": deepcopy(fiber),
                    "bubbles": deepcopy(case_cfg["bubbles"]),
                    "helium": helium,
                }
                run_cfg["bubbles"]["seed"] = geometry_seed

                generated = generate_structures(run_cfg)
                structure = generated["V"] if fill == 0.0 else generated["HeB"]
                stats = statistics(structure)

                structure_dir.mkdir(parents=True, exist_ok=True)
                write_lammps_data(data_path, structure, padding)
                write_yaml(
                    yml_path,
                    {
                        "potential": potential,
                        "case": case_name,
                        "state": state,
                        "geometry_seed": geometry_seed,
                        "thermal_seed": thermal_seed,
                        "fiber": deepcopy(fiber),
                        "bubbles": deepcopy(case_cfg["bubbles"]),
                        "helium": helium,
                        "statistics": stats,
                    },
                )

                rows.append(
                    make_row(
                        potential,
                        case_name,
                        state,
                        geometry_seed,
                        thermal_seed,
                        data_path,
                        stats,
                    )
                )

        with summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
