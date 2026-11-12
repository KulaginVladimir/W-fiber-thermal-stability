import numpy as np
from lammps import lammps

RC_WW = 3.7
SURFACE_OFFSET = 2.0
MASS_W = 183.84

_initialized = False
_max_id = 0
_he_emitted = None
_he_prev_d = None
_he_prev_xyz = None
_he_prev_vel = None
_he_prev_time = None
_clusters_file = None
_he_file = None
_he_active = True
_last_cluster_step = -1
_had_wrapping = None


def track_products(handle):
    lmp = lammps(ptr=handle)
    state = _gather_products(lmp)

    if lmp.extract_setting("world_rank") != 0:
        return

    _initialize(state)
    _write_clusters(lmp, state)
    if _he_active:
        _track_he(lmp, state)


def stop_he(handle):
    global _he_active
    _he_active = False


def _gather_products(lmp):
    natoms = int(lmp.get_natoms())

    ids = np.ctypeslib.as_array(
        lmp.gather_atoms_concat("id", 0, 1), shape=(natoms,)
    ).astype(np.int64, copy=True)
    types = np.ctypeslib.as_array(
        lmp.gather_atoms_concat("type", 0, 1), shape=(natoms,)
    ).astype(np.int32, copy=True)
    xyz = (
        np.ctypeslib.as_array(lmp.gather_atoms_concat("x", 1, 3), shape=(3 * natoms,))
        .reshape((natoms, 3))
        .copy()
    )
    vel = (
        np.ctypeslib.as_array(lmp.gather_atoms_concat("v", 1, 3), shape=(3 * natoms,))
        .reshape((natoms, 3))
        .copy()
    )
    heatflag = np.ctypeslib.as_array(
        lmp.gather_concat("i_heatflag", 0, 1), shape=(natoms,)
    ).astype(np.int32, copy=True)
    clid = np.ctypeslib.as_array(
        lmp.gather_concat("c_cl_source", 1, 1), shape=(natoms,)
    ).astype(np.int64, copy=True)

    return ids, types, xyz, vel, heatflag, clid


def _initialize(state):
    global _initialized, _max_id, _he_emitted
    global _he_prev_d, _he_prev_xyz, _he_prev_vel, _he_prev_time
    global _clusters_file, _he_file

    if _initialized:
        return

    ids = state[0]
    _max_id = int(ids.max()) if len(ids) else 0
    _he_emitted = np.zeros(_max_id + 1, dtype=np.bool_)
    _he_prev_d = np.full(_max_id + 1, np.nan, dtype=np.float64)
    _he_prev_xyz = np.full((_max_id + 1, 3), np.nan, dtype=np.float64)
    _he_prev_vel = np.full((_max_id + 1, 3), np.nan, dtype=np.float64)
    _he_prev_time = np.full(_max_id + 1, np.nan, dtype=np.float64)

    _clusters_file = open("w_clusters.txt", "w", 1)
    _clusters_file.write(
        "time_ps cluster_id N_W wrap_z x_com_A y_com_A z_com_A "
        "vx_com_Aps vy_com_Aps vz_com_Aps K_total_eV\n"
    )

    _he_file = open("he_release.txt", "w", 1)
    _he_file.write("time_ps atom_id x_A y_A z_A vx_Aps vy_Aps vz_Aps\n")

    _initialized = True


def _write_clusters(lmp, state):
    global _last_cluster_step, _had_wrapping

    ids, types, xyz, vel, heatflag, clid = state
    step = int(lmp.get_thermo("step"))
    wrapping_now = bool(np.any((types == 1) & (heatflag > 0)))
    transition = _had_wrapping is True and not wrapping_now
    due = _last_cluster_step < 0 or step - _last_cluster_step >= 1000

    _had_wrapping = wrapping_now
    if not due and not transition:
        return

    w = (types == 1) & (clid > 0)
    if not np.any(w):
        return

    w_clid = clid[w]
    unique, inverse = np.unique(w_clid, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)

    xcom = np.bincount(inverse, weights=xyz[w, 0]) / counts
    ycom = np.bincount(inverse, weights=xyz[w, 1]) / counts
    vxcom = np.bincount(inverse, weights=vel[w, 0]) / counts
    vycom = np.bincount(inverse, weights=vel[w, 1]) / counts
    vzcom = np.bincount(inverse, weights=vel[w, 2]) / counts

    boxlo, boxhi, *_ = lmp.extract_box()
    zlo = float(boxlo[2])
    lz = float(boxhi[2]) - zlo
    angle = 2.0 * np.pi * (xyz[w, 2] - zlo) / lz
    s = np.bincount(inverse, weights=np.sin(angle)) / counts
    c = np.bincount(inverse, weights=np.cos(angle)) / counts
    a = np.arctan2(s, c)
    a[a < 0.0] += 2.0 * np.pi
    zcom = zlo + lz * a / (2.0 * np.pi)

    mvv2e = float(lmp.extract_global("mvv2e"))
    v2 = np.sum(vel[w] * vel[w], axis=1)
    kinetic = 0.5 * MASS_W * mvv2e * np.bincount(inverse, weights=v2)

    wrapping = np.zeros(len(unique), dtype=np.int8)
    np.maximum.at(wrapping, inverse, (heatflag[w] > 0).astype(np.int8))

    time_ps = step * float(lmp.extract_global("dt"))

    for i in range(len(unique)):
        _clusters_file.write(
            f"{time_ps:.8f} {int(unique[i]):d} {int(counts[i]):d} "
            f"{int(wrapping[i]):d} "
            f"{xcom[i]:.8f} {ycom[i]:.8f} {zcom[i]:.8f} "
            f"{vxcom[i]:.8f} {vycom[i]:.8f} {vzcom[i]:.8f} "
            f"{kinetic[i]:.8f}\n"
        )

    _last_cluster_step = step


def _largest_cluster(types, clid):
    w = (types == 1) & (clid > 0)
    if not np.any(w):
        return None

    labels, counts = np.unique(clid[w], return_counts=True)
    return int(labels[np.argmax(counts)])


def _track_he(lmp, state):
    ids, types, xyz, vel, _, clid = state
    he_mask = types == 2
    if not np.any(he_mask):
        return

    lcc_id = _largest_cluster(types, clid)
    if lcc_id is None:
        return

    lcc = (types == 1) & (clid == lcc_id)
    he_ids_all = ids[he_mask]
    pending = ~_he_emitted[he_ids_all]
    if not np.any(pending):
        return

    he_ids = he_ids_all[pending]
    he_xyz = xyz[he_mask][pending]
    he_vel = vel[he_mask][pending]

    boxlo, boxhi, *_ = lmp.extract_box()
    zlo = float(boxlo[2])
    lz = float(boxhi[2]) - zlo
    time_ps = int(lmp.get_thermo("step")) * float(lmp.extract_global("dt"))

    d = _clearance_lcc(xyz[lcc], he_xyz, zlo, lz)
    prev_d = _he_prev_d[he_ids]
    crossing = np.isfinite(prev_d) & (prev_d <= 0.0) & (d > 0.0)

    for j in np.nonzero(crossing)[0]:
        atom_id = int(he_ids[j])
        d0 = float(prev_d[j])
        d1 = float(d[j])
        f = -d0 / (d1 - d0)
        f = min(1.0, max(0.0, f))

        t0 = float(_he_prev_time[atom_id])
        p0 = _he_prev_xyz[atom_id].copy()
        p1 = he_xyz[j]
        v0 = _he_prev_vel[atom_id]
        v1 = he_vel[j]

        dz = p1[2] - p0[2]
        if dz > 0.5 * lz:
            dz -= lz
        elif dz < -0.5 * lz:
            dz += lz

        p = p0 + f * (p1 - p0)
        p[2] = zlo + np.mod((p0[2] - zlo) + f * dz, lz)
        v = v0 + f * (v1 - v0)
        t = t0 + f * (time_ps - t0)

        _he_file.write(
            f"{t:.8f} {atom_id:d} "
            f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
            f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n"
        )
        _he_emitted[atom_id] = True

    keep = ~_he_emitted[he_ids]
    keep_ids = he_ids[keep]
    _he_prev_d[keep_ids] = d[keep]
    _he_prev_xyz[keep_ids] = he_xyz[keep]
    _he_prev_vel[keep_ids] = he_vel[keep]
    _he_prev_time[keep_ids] = time_ps


def _clearance_lcc(w_xyz, he_xyz, zlo, lz):
    nhe = len(he_xyz)
    out = np.full(nhe, -np.inf, dtype=np.float64)
    if nhe == 0 or len(w_xyz) == 0:
        return out

    nbins = max(1, int(np.floor(lz / RC_WW)))
    width = lz / nbins

    wz = np.mod(w_xyz[:, 2] - zlo, lz)
    hz = np.mod(he_xyz[:, 2] - zlo, lz)
    wb = np.floor(wz / width).astype(np.int32)
    hb = np.floor(hz / width).astype(np.int32)
    wb[wb == nbins] = nbins - 1
    hb[hb == nbins] = nbins - 1

    w_by_bin = [np.nonzero(wb == b)[0] for b in range(nbins)]

    for b in np.unique(hb):
        hi = np.nonzero(hb == b)[0]
        bins = [(int(b) - 1) % nbins, int(b) % nbins, (int(b) + 1) % nbins]
        wi = np.unique(np.concatenate([w_by_bin[k] for k in bins]))
        if len(wi) == 0:
            continue

        w = w_xyz[wi]
        h = he_xyz[hi]

        dz = np.abs(w[:, None, 2] - h[None, :, 2])
        dz = np.minimum(dz, lz - dz)
        local = dz <= RC_WW
        counts = np.sum(local, axis=0)
        valid = counts > 0
        if not np.any(valid):
            continue

        centers = np.zeros((len(hi), 2), dtype=np.float64)
        centers[valid] = local[:, valid].T @ w[:, :2]
        centers[valid] /= counts[valid, None]

        radial = h[:, :2] - centers
        rnorm = np.linalg.norm(radial, axis=1)
        valid &= rnorm > 1.0e-12
        if not np.any(valid):
            continue

        u = np.zeros_like(radial)
        u[valid] = radial[valid] / rnorm[valid, None]

        proj = w[:, :2] @ u.T
        proj[~local] = -np.inf
        qsurf = np.max(proj, axis=0)
        qhe = np.sum(h[:, :2] * u, axis=1)

        d = qhe - qsurf - SURFACE_OFFSET
        out[hi[valid]] = d[valid]

    return out
