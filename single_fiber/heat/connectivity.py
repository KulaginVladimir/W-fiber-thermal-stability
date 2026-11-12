import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from lammps import lammps

RC = 3.7
CHECK_EVERY = 500

_last_step = -1
_last_value = 1.0


def _wrapping_mask_z(xyz, boxlo, boxhi):
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    lz = float(boxhi[2] - boxlo[2])
    xymin = np.min(xyz[:, :2], axis=0)
    spans = np.ptp(xyz[:, :2], axis=0)
    points = np.empty_like(xyz)
    points[:, :2] = xyz[:, :2] - xymin + RC
    points[:, 2] = np.mod(xyz[:, 2] - boxlo[2], lz)
    size = np.array([2 * spans[0] + 4 * RC, 2 * spans[1] + 4 * RC, lz])

    tree = cKDTree(points, boxsize=size)
    pairs = tree.query_pairs(RC, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(n, dtype=np.bool_)

    a, b = pairs[:, 0], pairs[:, 1]
    delta = points[b] - points[a]
    dz = delta[:, 2]
    delta[:, 2] = np.where(dz > lz / 2, dz - lz, np.where(dz < -lz / 2, dz + lz, dz))
    pairs = pairs[np.einsum("ij,ij->i", delta, delta) < RC * RC]
    if len(pairs) == 0:
        return np.zeros(n, dtype=np.bool_)

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
    return wraps[labels]


def check_connectivity_z(handle):
    global _last_step, _last_value

    lmp = lammps(ptr=handle)
    step = int(lmp.get_thermo("step"))
    if _last_step >= 0 and step - _last_step < CHECK_EVERY:
        return _last_value

    comm = lmp.get_mpi_comm()
    if comm is None:
        raise RuntimeError("MPI communicator unavailable")
    rank = comm.Get_rank()
    natoms = int(lmp.get_natoms())

    ids_ptr = lmp.gather_atoms_concat("id", 0, 1)
    types_ptr = lmp.gather_atoms_concat("type", 0, 1)
    xyz_ptr = lmp.gather_atoms_concat("x", 1, 3)

    if rank == 0:
        ids = np.ctypeslib.as_array(ids_ptr, shape=(natoms,)).copy()
        types = np.ctypeslib.as_array(types_ptr, shape=(natoms,)).copy()
        xyz = (
            np.ctypeslib.as_array(xyz_ptr, shape=(3 * natoms,))
            .reshape(natoms, 3)
            .copy()
        )
        mask = types == 1
        boxlo, boxhi, *_ = lmp.extract_box()
        wrapping = _wrapping_mask_z(xyz[mask], boxlo, boxhi)
        selected = np.zeros(int(ids.max()) + 1 if natoms else 1, dtype=np.uint8)
        selected[ids[mask][wrapping]] = 1
        value = float(np.any(wrapping))
    else:
        selected = None
        value = None

    count = comm.bcast(len(selected) if rank == 0 else None, root=0)
    if rank != 0:
        selected = np.empty(count, dtype=np.uint8)
    comm.Bcast(selected, root=0)
    value = comm.bcast(value, root=0)

    nlocal = int(lmp.extract_setting("nlocal"))
    local_ids = np.ctypeslib.as_array(lmp.extract_atom("id"), shape=(nlocal,))
    heatflag = np.ctypeslib.as_array(lmp.extract_atom("i_heatflag"), shape=(nlocal,))
    heatflag[:] = selected[local_ids]

    _last_step = step
    _last_value = value
    return value
