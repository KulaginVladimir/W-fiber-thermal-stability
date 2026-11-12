import math
from collections import defaultdict

import numpy as np


class Structure:
    def __init__(
        self,
        w,
        he,
        radius,
        length,
        w_initial,
        bubbles,
        removed_per_bubble,
    ):
        self.w = w
        self.he = he
        self.radius = radius
        self.length = length
        self.w_initial = w_initial
        self.bubbles = bubbles
        self.removed_per_bubble = removed_per_bubble


def _bcc_cylinder(a: float, radius_lattice: float, length_cells: int):
    radius = radius_lattice * a
    length = length_cells * a
    nxy = math.ceil(radius / a) + 1
    basis = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))

    atoms = []
    r2 = radius * radius
    for i in range(-nxy, nxy + 1):
        for j in range(-nxy, nxy + 1):
            for k in range(length_cells):
                for bx, by, bz in basis:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    if x * x + y * y <= r2:
                        atoms.append((x, y, z))

    return np.asarray(atoms, dtype=float), radius, length


def _draw_truncated_normal(rng, mean, std, low, high):
    while True:
        x = rng.normal(mean, std)
        if low <= x <= high:
            return x


def _bubble_diameters(cfg, radius, length, rng):
    spec = cfg["diameter_fraction"]
    fiber_diameter = 2.0 * radius

    mean = float(spec["mean"]) * fiber_diameter
    std = float(spec["std"]) * fiber_diameter
    low = float(spec["min"]) * fiber_diameter
    high = float(spec["max"]) * fiber_diameter

    if not (0.0 < low <= mean <= high):
        raise ValueError("Require 0 < min <= mean <= max for bubble diameter.")
    if std <= 0.0:
        raise ValueError("Bubble diameter std must be positive.")
    if high >= length:
        raise ValueError(
            "Largest bubble diameter must be smaller than periodic length."
        )

    target = float(cfg["void_fraction"]) * math.pi * radius**2 * length
    diameters = []
    volume = 0.0

    while volume < target:
        d = _draw_truncated_normal(rng, mean, std, low, high)
        diameters.append(d)
        volume += math.pi * d**3 / 6.0

    if len(diameters) > 1:
        last = math.pi * diameters[-1] ** 3 / 6.0
        if abs((volume - last) - target) < abs(volume - target):
            diameters.pop()

    return np.sort(np.asarray(diameters, dtype=float))[::-1]


def _periodic_dz(dz, length):
    return dz - length * np.rint(dz / length)


def _place_bubbles(
    diameters,
    radius,
    length,
    rng,
    min_surface_ligament_ratio,
):
    bubbles = []

    if min_surface_ligament_ratio < 0.0:
        raise ValueError("min_surface_ligament_ratio must be non-negative.")

    for d in diameters:
        r = 0.5 * d

        rho_max = radius - (1.0 + min_surface_ligament_ratio) * r

        if rho_max <= 0.0:
            raise ValueError("Bubble is too large for the requested surface ligament.")

        for _ in range(100000):
            rho = rho_max * math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            z = length * rng.random()

            ok = True
            for xj, yj, zj, rj in bubbles:
                dx = x - xj
                dy = y - yj
                dz = _periodic_dz(z - zj, length)

                if dx * dx + dy * dy + dz * dz < (r + rj) ** 2:
                    ok = False
                    break

            if ok:
                bubbles.append((x, y, z, r))
                break
        else:
            raise RuntimeError(
                "Could not place all bubbles with the requested surface ligament."
            )

    return np.asarray(bubbles, dtype=float)


def _remove_w(w0, bubbles, length):
    removed = np.zeros(len(w0), dtype=bool)
    counts = []

    for x, y, z, r in bubbles:
        dx = w0[:, 0] - x
        dy = w0[:, 1] - y
        dz = _periodic_dz(w0[:, 2] - z, length)
        inside = dx * dx + dy * dy + dz * dz <= r * r
        counts.append(int(np.count_nonzero(inside)))
        removed |= inside

    return w0[~removed].copy(), np.asarray(counts, dtype=int)


class _CellList:
    def __init__(self, points, cutoff, length):
        self.cutoff = float(cutoff)
        self.cutoff2 = self.cutoff * self.cutoff
        self.length = float(length)
        self.nz = max(1, math.ceil(self.length / self.cutoff))
        self.cells = defaultdict(list)
        for p in points:
            self.add(p)

    def _key(self, p):
        return (
            math.floor(p[0] / self.cutoff),
            math.floor(p[1] / self.cutoff),
            math.floor(p[2] / self.cutoff) % self.nz,
        )

    def add(self, p):
        self.cells[self._key(p)].append(np.asarray(p, dtype=float))

    def too_close(self, p):
        ix, iy, iz = self._key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz_cell in (-1, 0, 1):
                    key = (ix + dx, iy + dy, (iz + dz_cell) % self.nz)
                    for q in self.cells.get(key, ()):
                        dxp = p[0] - q[0]
                        dyp = p[1] - q[1]
                        dzp = _periodic_dz(p[2] - q[2], self.length)
                        if dxp * dxp + dyp * dyp + dzp * dzp < self.cutoff2:
                            return True
        return False


def _random_point_in_sphere(rng, center, radius, length):
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    p = center + direction * radius * rng.random() ** (1.0 / 3.0)
    p[2] %= length
    return p


def _fill_he(w, bubbles, removed_per_bubble, cfg, a, length):
    ratio = float(cfg["he_per_vacancy"])
    if ratio < 0.0:
        raise ValueError("he_per_vacancy must be non-negative.")

    min_dist = float(cfg["min_distance_lattice"]) * a
    if min_dist <= 0.0:
        raise ValueError("min_distance_lattice must be positive.")

    rng = np.random.default_rng(int(cfg["seed"]))

    w_cells = _CellList(w, min_dist, length)
    he_cells = _CellList(np.empty((0, 3)), min_dist, length)
    he = []

    for bubble, n_removed in zip(bubbles, removed_per_bubble):
        n_he = round(ratio * n_removed)
        center = bubble[:3]
        radius = bubble[3]

        for _ in range(n_he):
            for _ in range(20000):
                p = _random_point_in_sphere(rng, center, radius, length)
                if w_cells.too_close(p) or he_cells.too_close(p):
                    continue
                he.append(p)
                he_cells.add(p)
                break
            else:
                raise RuntimeError(
                    "Could not place He atoms with the requested minimum distance."
                )

    return np.asarray(he, dtype=float).reshape((-1, 3))


def generate_structures(cfg):
    fiber = cfg["fiber"]
    a = float(fiber["lattice_parameter_angstrom"])
    radius_lattice = float(fiber["radius_lattice"])
    length_cells = int(fiber["length_cells"])

    if a <= 0.0:
        raise ValueError("lattice_parameter_angstrom must be positive.")

    w0, radius, length = _bcc_cylinder(a, radius_lattice, length_cells)

    bubble_cfg = cfg["bubbles"]
    rng = np.random.default_rng(int(bubble_cfg["seed"]))
    diameters = _bubble_diameters(bubble_cfg, radius, length, rng)
    min_surface_ligament_ratio = float(bubble_cfg["min_surface_ligament_ratio"])

    bubbles = _place_bubbles(
        diameters,
        radius,
        length,
        rng,
        min_surface_ligament_ratio,
    )

    w_def, removed = _remove_w(w0, bubbles, length)

    empty = np.empty((0, 3), dtype=float)
    no_bubbles = np.empty((0, 4), dtype=float)
    no_removed = np.empty(0, dtype=int)

    pure = Structure(w0, empty, radius, length, len(w0), no_bubbles, no_removed)
    void = Structure(w_def, empty, radius, length, len(w0), bubbles, removed)

    he = _fill_he(
        w_def,
        bubbles,
        removed,
        cfg["helium"],
        a,
        length,
    )
    heb = Structure(w_def.copy(), he, radius, length, len(w0), bubbles, removed)

    if not np.array_equal(void.w, heb.w):
        raise RuntimeError("V and HeB tungsten structures differ.")

    return {"W": pure, "V": void, "HeB": heb}


def statistics(structure):
    n_removed = structure.w_initial - len(structure.w)
    bubble_d = 2.0 * structure.bubbles[:, 3] if len(structure.bubbles) else np.empty(0)

    stats = {
        "w_initial": int(structure.w_initial),
        "w_final": len(structure.w),
        "w_removed": n_removed,
        "void_fraction": float(n_removed / structure.w_initial)
        if structure.w_initial
        else 0.0,
        "he_atoms": len(structure.he),
        "he_to_w_ratio": float(len(structure.he) / len(structure.w))
        if len(structure.w)
        else 0.0,
        "he_atomic_fraction": float(
            len(structure.he) / (len(structure.w) + len(structure.he))
        )
        if (len(structure.w) + len(structure.he))
        else 0.0,
        "he_per_vacancy": float(len(structure.he) / n_removed) if n_removed else 0.0,
        "bubble_count": len(structure.bubbles),
    }

    if len(bubble_d):
        stats["bubble_diameter_angstrom"] = {
            "mean": float(np.mean(bubble_d)),
            "std": float(np.std(bubble_d)),
            "min": float(np.min(bubble_d)),
            "max": float(np.max(bubble_d)),
        }

    if len(structure.bubbles):
        rho = np.sqrt(structure.bubbles[:, 0] ** 2 + structure.bubbles[:, 1] ** 2)
        bubble_r = structure.bubbles[:, 3]

        surface_ligament = structure.radius - rho - bubble_r
        surface_ligament_ratio = surface_ligament / bubble_r

        stats["surface_ligament_angstrom"] = {
            "mean": float(np.mean(surface_ligament)),
            "min": float(np.min(surface_ligament)),
        }

        stats["surface_ligament_ratio"] = {
            "mean": float(np.mean(surface_ligament_ratio)),
            "min": float(np.min(surface_ligament_ratio)),
        }

    return stats
