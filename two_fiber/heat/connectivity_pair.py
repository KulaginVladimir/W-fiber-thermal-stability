import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from lammps import lammps

RC = 3.7
CHECK_EVERY = 500

_last_source_step = -1
_last_source_value = 1.0
_source_active = None
_source_mid = None
_source_dir = None
_source_broken = False


def _components_z(xyz, boxlo, boxhi):
    n = len(xyz)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.zeros(0, dtype=np.bool_)

    lz = float(boxhi[2] - boxlo[2])
    xymin = np.min(xyz[:, :2], axis=0)
    spans = np.ptp(xyz[:, :2], axis=0)
    points = np.empty_like(xyz)
    points[:, :2] = xyz[:, :2] - xymin + RC
    points[:, 2] = np.mod(xyz[:, 2] - boxlo[2], lz)
    size = np.array([2 * spans[0] + 4 * RC, 2 * spans[1] + 4 * RC, lz])

    pairs = cKDTree(points, boxsize=size).query_pairs(RC, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.bool_)

    a, b = pairs[:, 0], pairs[:, 1]
    delta = points[b] - points[a]
    dz = delta[:, 2]
    delta[:, 2] = np.where(dz > lz / 2, dz - lz, np.where(dz < -lz / 2, dz + lz, dz))
    pairs = pairs[np.einsum("ij,ij->i", delta, delta) < RC * RC]
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.bool_)

    a, b = pairs[:, 0], pairs[:, 1]
    dz = xyz[b, 2] - xyz[a, 2]
    shift = np.where(dz > lz / 2, -1, np.where(dz < -lz / 2, 1, 0)).astype(np.int32)
    edges = csr_matrix(
        (np.ones(2 * len(a), dtype=np.int8), (np.r_[a, b], np.r_[b, a])), shape=(n, n)
    )
    forest = minimum_spanning_tree(edges).tocoo()
    u, v = forest.row, forest.col
    dz = xyz[v, 2] - xyz[u, 2]
    s = np.where(dz > lz / 2, -1, np.where(dz < -lz / 2, 1, 0)).astype(np.int32)
    links = csr_matrix((np.r_[s, -s], (np.r_[u, v], np.r_[v, u])), shape=(n, n))

    visited = np.zeros(n, dtype=np.bool_)
    images = np.zeros(n, dtype=np.int32)
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for k in range(links.indptr[i], links.indptr[i + 1]):
                j = links.indices[k]
                if not visited[j]:
                    visited[j] = True
                    images[j] = images[i] + links.data[k]
                    stack.append(j)

    count, labels = connected_components(edges, directed=False)
    wraps = np.zeros(count, dtype=np.bool_)
    inconsistent = images[b] - images[a] != shift
    wraps[labels[a[inconsistent]]] = True
    return labels, wraps


def _gather(lmp):
    comm = lmp.get_mpi_comm()
    rank = comm.Get_rank()
    natoms = int(lmp.get_natoms())
    ids_ptr = lmp.gather_atoms_concat("id", 0, 1)
    types_ptr = lmp.gather_atoms_concat("type", 0, 1)
    xyz_ptr = lmp.gather_atoms_concat("x", 1, 3)
    if rank != 0:
        return comm, None, None, None, None, None
    ids = np.ctypeslib.as_array(ids_ptr, shape=(natoms,)).copy()
    types = np.ctypeslib.as_array(types_ptr, shape=(natoms,)).copy()
    xyz = np.ctypeslib.as_array(xyz_ptr, shape=(3 * natoms,)).reshape(natoms, 3).copy()
    boxlo, boxhi, *_ = lmp.extract_box()
    return comm, ids, types, xyz, boxlo, boxhi


def check_connectivity_source(handle):
    global \
        _last_source_step, \
        _last_source_value, \
        _source_active, \
        _source_mid, \
        _source_dir, \
        _source_broken
    if _source_broken:
        return 0.0

    lmp = lammps(ptr=handle)
    step = int(lmp.get_thermo("step"))
    if _last_source_step >= 0 and step - _last_source_step < CHECK_EVERY:
        return _last_source_value

    comm, ids, types, xyz, boxlo, boxhi = _gather(lmp)
    if comm.Get_rank() == 0:
        if _source_active is None:
            _source_active = np.zeros(int(ids.max()) + 1, dtype=np.bool_)
            _source_active[ids[types == 1]] = True
            source_xyz = xyz[types == 1, :2]
            target_xyz = xyz[types == 3, :2]
            if len(source_xyz) and len(target_xyz):
                source_xy = source_xyz.mean(axis=0)
                target_xy = target_xyz.mean(axis=0)
                _source_mid = (source_xy + target_xy) / 2
                _source_dir = target_xy - source_xy
            else:
                _source_mid = np.zeros(2)
                _source_dir = np.zeros(2)
        mask = types == 1
        source_ids = ids[mask]
        labels, wraps = _components_z(xyz[mask], boxlo, boxhi)
        source_side = ((xyz[mask, :2] - _source_mid) @ _source_dir) < 0
        selected_ids = np.empty(0, dtype=np.int64)
        if len(wraps):
            overlap = np.bincount(
                labels,
                weights=(_source_active[source_ids] & source_side).astype(np.int32),
                minlength=len(wraps),
            )
            root = int(np.argmax(overlap))
            keep = labels == root
            if (
                overlap[root] > 0
                and wraps[root]
                and 2 * np.count_nonzero(source_side[keep]) > np.count_nonzero(keep)
            ):
                selected_ids = source_ids[keep]
        _source_active[:] = False
        _source_active[selected_ids] = True
        selected = np.zeros(len(_source_active), dtype=np.uint8)
        selected[selected_ids] = 1
        value = float(len(selected_ids) > 0)
    else:
        selected = None
        value = None

    count = comm.bcast(len(selected) if comm.Get_rank() == 0 else None, root=0)
    if comm.Get_rank() != 0:
        selected = np.empty(count, dtype=np.uint8)
    comm.Bcast(selected, root=0)
    value = comm.bcast(value, root=0)

    nlocal = int(lmp.extract_setting("nlocal"))
    local_ids = np.ctypeslib.as_array(lmp.extract_atom("id"), shape=(nlocal,))
    heatflag = np.ctypeslib.as_array(lmp.extract_atom("i_heatflag"), shape=(nlocal,))
    heatflag[:] = selected[local_ids]

    _source_broken = value == 0.0
    _last_source_step = step
    _last_source_value = value
    return value
