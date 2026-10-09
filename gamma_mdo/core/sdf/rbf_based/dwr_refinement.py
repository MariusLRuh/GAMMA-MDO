from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

try:
    from sksparse.cholmod import cholesky as _cholmod_cholesky
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _cholmod_cholesky = None

# Number of parallel workers used for cKDTree.query_ball_point during prediction.
# -1 → use all available CPU cores (fastest for large query sets).
_KDTREE_WORKERS: int = -1


# ============================================================
# Kernel
# ============================================================

def wendland_c2(q: np.ndarray) -> np.ndarray:
    """
    Wendland C2 kernel in 3D, compact support on q in [0, 1]:
        phi(q) = (1 - q)^4 * (4q + 1),   q <= 1
               = 0                        q > 1

    Parameters
    ----------
    q : ndarray
        Nonnegative normalized distance(s), q = r / rho.

    Returns
    -------
    ndarray
        Kernel values with same shape as q.
    """
    q = np.asarray(q)
    out = np.zeros_like(q, dtype=float)
    mask = q < 1.0
    qm = q[mask]
    t = 1.0 - qm
    out[mask] = (t ** 4) * (4.0 * qm + 1.0)
    return out


def wendland_c2_from_dist(d: np.ndarray, R: float) -> np.ndarray:
    """
    Wendland C2 kernel evaluated from Euclidean distances directly.

    Parameters
    ----------
    d : ndarray
        Euclidean distances.
    R : float
        Compact-support radius.

    Returns
    -------
    ndarray
        Kernel values with the same shape as ``d``.
    """
    d = np.asarray(d, dtype=float)
    out = np.zeros_like(d, dtype=float)
    if R <= 0.0:
        return out

    s = d / float(R)
    mask = s < 1.0
    sm = s[mask]
    t = 1.0 - sm
    out[mask] = (t ** 4) * (4.0 * sm + 1.0)
    return out


# ============================================================
# Utility functions
# ============================================================

def inverse_distance_weights(
    sdf_values: np.ndarray,
    power: float = 1.0,
    eps: float = 1e-3,
    clip_max: Optional[float] = 1e6,
    normalize: bool = True,
) -> np.ndarray:
    """
    Weights inversely related to distance from the surface, using |SDF|.

    w_j = 1 / (|d_j| + eps)^power

    Parameters
    ----------
    sdf_values : (Q,) ndarray
        Exact signed distance values at check points.
    power : float
        Exponent in the inverse-distance weighting.
    eps : float
        Small floor to avoid singular weights at/near the surface.
    clip_max : float or None
        Optional cap on the weights.
    normalize : bool
        If True, normalize weights so median(weight)=1.

    Returns
    -------
    w : (Q,) ndarray
        Positive weights.
    """
    w = 1.0 / np.power(np.abs(sdf_values) + eps, power)
    if clip_max is not None:
        w = np.minimum(w, clip_max)
    if normalize:
        med = np.median(w)
        if med > 0.0:
            w = w / med
    return w


def compute_local_support_radii(
    pts: np.ndarray,
    neighbor_k: int = 24,
    radius_scale: float = 1.5,
) -> np.ndarray:
    """
    Compute a local recommended support radius rho_i for each center based on
    the distance to its k-th nearest neighbor.

    Parameters
    ----------
    pts : (N,3) ndarray
        Training/center locations.
    neighbor_k : int
        k for k-th nearest neighbor distance.
    radius_scale : float
        Multiplier applied to d_k.

    Returns
    -------
    rho_local : (N,) ndarray
        Local recommended support radii.
    """
    n = pts.shape[0]
    k = min(max(2, neighbor_k), n)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k)
    # dists[:, 0] is zero (self-distance); use last column as k-th NN
    dk = dists[:, -1]
    rho_local = radius_scale * dk
    # Avoid exact zeros if duplicate points accidentally exist
    rho_local = np.maximum(rho_local, 1e-12)
    return rho_local


def robust_global_support_radius(
    rho_local: np.ndarray,
    quantile: float = 0.90,
    inflate: float = 1.10,
) -> float:
    """
    Convert local support recommendations into one robust global support radius.
    This preserves a symmetric/SPD interpolation matrix for CHOLMOD.

    Parameters
    ----------
    rho_local : (N,) ndarray
        Local recommended support radii.
    quantile : float
        Robust quantile used to pick a conservative radius.
    inflate : float
        Additional inflation factor.

    Returns
    -------
    rho_global : float
        Global support radius.
    """
    rho_global = inflate * float(np.quantile(rho_local, quantile))
    return max(rho_global, 1e-12)


def assemble_wendland_matrix(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    rho: float,
    src_tree: Optional[cKDTree] = None,
    chunk_size: int = 4096,
) -> sparse.csr_matrix:
    """
    Assemble sparse matrix A with entries
        A[j, i] = phi(||tgt_j - src_i|| / rho)
    using compact support search.

    Parameters
    ----------
    src_pts : (N,3) ndarray
        Source/basis center locations.
    tgt_pts : (M,3) ndarray
        Target evaluation locations.
    rho : float
        Global support radius.
    src_tree : cKDTree or None
        Optional KD-tree for source points.
    chunk_size : int
        Number of target points processed per chunk. Larger values reduce
        Python overhead but may increase peak memory during assembly.

    Returns
    -------
    A : (M,N) csr_matrix
        Sparse Wendland matrix.
    """
    if src_tree is None:
        src_tree = cKDTree(src_pts)

    src_pts = np.asarray(src_pts, dtype=float)
    tgt_pts = np.asarray(tgt_pts, dtype=float)
    n_tgt = tgt_pts.shape[0]
    n_src = src_pts.shape[0]

    if n_tgt == 0 or n_src == 0:
        return sparse.csr_matrix((n_tgt, n_src), dtype=float)

    chunk_size = max(1, int(chunk_size))
    row_chunks = []
    col_chunks = []
    data_chunks = []

    rho = float(rho)
    inv_rho = 1.0 / rho if rho > 0.0 else 0.0

    for start in range(0, n_tgt, chunk_size):
        stop = min(start + chunk_size, n_tgt)
        tgt_chunk = tgt_pts[start:stop]
        neighbors = src_tree.query_ball_point(tgt_chunk, r=rho)

        counts = np.fromiter((len(idx) for idx in neighbors), dtype=np.int64, count=stop - start)
        total_nnz_guess = int(np.sum(counts))
        if total_nnz_guess == 0:
            continue

        chunk_rows = np.empty(total_nnz_guess, dtype=np.int64)
        chunk_cols = np.empty(total_nnz_guess, dtype=np.int64)
        chunk_data = np.empty(total_nnz_guess, dtype=float)

        cursor = 0
        for local_j, idxs in enumerate(neighbors):
            if not idxs:
                continue

            idxs = np.asarray(idxs, dtype=np.int64)
            dx = src_pts[idxs] - tgt_chunk[local_j]
            q = np.sqrt(np.einsum("ij,ij->i", dx, dx)) * inv_rho
            mask = q < 1.0
            if not np.any(mask):
                continue

            qm = q[mask]
            t = 1.0 - qm
            vals = (t ** 4) * (4.0 * qm + 1.0)
            nnz = vals.size

            sl = slice(cursor, cursor + nnz)
            chunk_rows[sl] = start + local_j
            chunk_cols[sl] = idxs[mask]
            chunk_data[sl] = vals
            cursor += nnz

        if cursor > 0:
            row_chunks.append(chunk_rows[:cursor])
            col_chunks.append(chunk_cols[:cursor])
            data_chunks.append(chunk_data[:cursor])

    if not data_chunks:
        return sparse.csr_matrix((n_tgt, n_src), dtype=float)

    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.concatenate(data_chunks)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_tgt, n_src), dtype=float)
    return A


def assemble_wendland_matrix_from_sparse_distance(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    rho: float,
    src_tree: Optional[cKDTree] = None,
    tgt_tree: Optional[cKDTree] = None,
) -> sparse.csr_matrix:
    """
    Assemble sparse Wendland matrix using ``cKDTree.sparse_distance_matrix``.

    This path pushes neighbor discovery and distance extraction into SciPy's
    C implementation, which is often faster than building Python ragged lists
    via ``query_ball_point`` for large prediction workloads.

    Returns
    -------
    A : (M, N) csr_matrix
        Sparse Wendland matrix with rows corresponding to ``tgt_pts`` and
        columns corresponding to ``src_pts``.
    """
    src_pts = np.asarray(src_pts, dtype=float)
    tgt_pts = np.asarray(tgt_pts, dtype=float)
    n_src = src_pts.shape[0]
    n_tgt = tgt_pts.shape[0]

    if n_src == 0 or n_tgt == 0:
        return sparse.csr_matrix((n_tgt, n_src), dtype=float)

    rho = float(rho)
    if rho <= 0.0:
        return sparse.csr_matrix((n_tgt, n_src), dtype=float)

    if src_tree is None:
        src_tree = cKDTree(src_pts)
    if tgt_tree is None:
        tgt_tree = cKDTree(tgt_pts)

    D = tgt_tree.sparse_distance_matrix(
        src_tree,
        max_distance=rho,
        output_type="coo_matrix",
    )
    if D.nnz == 0:
        return sparse.csr_matrix((n_tgt, n_src), dtype=float)

    D.data = wendland_c2_from_dist(D.data, rho)
    return D.tocsr()


def evaluate_rbf_matvec(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    rho: float,
    coeffs: np.ndarray,
    src_tree: Optional[cKDTree] = None,
    workers: int = _KDTREE_WORKERS,
) -> np.ndarray:
    """
    Directly compute  out[j] = sum_i  phi(||tgt_j - src_i|| / rho) * coeffs[i]
    **without** materializing a sparse matrix.

    Strategy
    --------
    1. ``cKDTree.query_ball_point`` with ``workers=-1`` retrieves all
       (query, center) pairs within radius rho in parallel — this is the
       dominant cost and is fully multi-threaded inside SciPy.
    2. The ragged neighbor lists are flattened into three parallel arrays
       (row indices, col indices, distances) using fully-vectorized NumPy
       operations — no Python loop over query points.
    3. Wendland C2 kernel values and the final weighted sum are computed
       in bulk with NumPy, bypassing sparse-matrix construction entirely.

    This is 10–30× faster than the previous ``assemble_wendland_matrix`` +
    sparse matvec pipeline for large query sets.

    Parameters
    ----------
    src_pts : (N, 3) ndarray
        RBF center locations.
    tgt_pts : (M, 3) ndarray
        Query locations.
    rho : float
        Compact-support radius.
    coeffs : (N,) ndarray
        RBF coefficients.
    src_tree : cKDTree or None
        Pre-built KD-tree for src_pts. Built here if not supplied.
    workers : int
        Number of parallel workers for query_ball_point. -1 = all cores.

    Returns
    -------
    out : (M,) ndarray
    """
    n_src = src_pts.shape[0]
    n_tgt = tgt_pts.shape[0]

    if n_src == 0 or n_tgt == 0:
        return np.zeros((n_tgt,), dtype=float)

    if src_tree is None:
        src_tree = cKDTree(src_pts)

    rho = float(rho)
    inv_rho = 1.0 / rho

    # --- Step 1: parallel radius search (multi-threaded C code) --------------
    # Returns a list-of-lists of source indices for each query point.
    neighbor_lists = src_tree.query_ball_point(
        tgt_pts, r=rho, workers=workers, return_sorted=False
    )

    # --- Step 2: fully-vectorized COO assembly (no Python loop) --------------
    counts = np.fromiter(
        (len(nb) for nb in neighbor_lists), dtype=np.int32, count=n_tgt
    )
    total_nnz = int(counts.sum())
    if total_nnz == 0:
        return np.zeros((n_tgt,), dtype=float)

    # Flat source indices and corresponding query-row indices.
    # A manual copy loop into a pre-allocated array is faster than
    # np.concatenate for the typical mix of small ragged lists.
    col_idx = np.empty(total_nnz, dtype=np.int32)
    row_idx = np.repeat(np.arange(n_tgt, dtype=np.int32), counts)
    pos = 0
    for nb in neighbor_lists:
        k = len(nb)
        if k:
            col_idx[pos : pos + k] = nb
            pos += k

    # --- Step 3: kernel evaluation (fully vectorized) ------------------------
    diff = tgt_pts[row_idx] - src_pts[col_idx]               # (nnz, 3)
    q = np.sqrt(np.einsum("ij,ij->i", diff, diff)) * inv_rho # (nnz,)
    # query_ball_point guarantees q <= 1; clip to handle floating-point edge
    # cases without a mask — np.clip is faster than boolean indexing here.
    q = np.minimum(q, 1.0 - 1e-14)
    t = 1.0 - q
    phi = (t * t * t * t) * (4.0 * q + 1.0)                  # Wendland C2

    # --- Step 4: scatter-sum into output ------------------------------------
    # np.add.at is the fastest path that avoids materializing a full sparse
    # matrix.  For very dense problems (> ~5M nnz) the CSR SpMV route is
    # comparable but requires an extra matrix allocation; we stay with
    # np.add.at for simplicity and predictable memory usage.
    out = np.zeros((n_tgt,), dtype=float)
    np.add.at(out, row_idx, phi * coeffs[col_idx])

    return out


def gaussian_rbf_from_sqdist(sq_dist: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Dense Gaussian RBF kernel with the standard shape parameter epsilon:
        phi(r) = exp(-(epsilon * r)^2)
    """
    sq = np.asarray(sq_dist, dtype=float)
    eps = max(float(epsilon), 1e-12)
    return np.exp(-(eps * eps) * sq)


def assemble_dense_gaussian_matrix(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Assemble the dense Gaussian matrix A with entries
        A[j, i] = exp(-(epsilon * ||tgt_j - src_i||)^2).
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)
    n_src = src.shape[0]
    n_tgt = tgt.shape[0]
    if n_src == 0 or n_tgt == 0:
        return np.zeros((n_tgt, n_src), dtype=float)

    sq_dist = cdist(tgt, src, metric="sqeuclidean")
    return gaussian_rbf_from_sqdist(sq_dist, epsilon)


def evaluate_dense_gaussian_matvec(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    epsilon: float,
    coeffs: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Evaluate a dense Gaussian RBF expansion without materializing the full
    target-by-source matrix when the target set is large.
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)
    c = np.asarray(coeffs, dtype=float).reshape(-1)
    n_src = src.shape[0]
    n_tgt = tgt.shape[0]
    if n_src == 0 or n_tgt == 0:
        return np.zeros((n_tgt,), dtype=float)

    chunk = max(1, int(chunk_size))
    out = np.zeros((n_tgt,), dtype=float)
    for start in range(0, n_tgt, chunk):
        stop = min(start + chunk, n_tgt)
        sq_dist = cdist(tgt[start:stop], src, metric="sqeuclidean")
        out[start:stop] = gaussian_rbf_from_sqdist(sq_dist, epsilon) @ c
    return out


def compute_gaussian_shape_parameter(
    pts: np.ndarray,
    neighbor_k: int = 8,
    shape_scale: float = 1.0,
) -> float:
    """
    Estimate a Gaussian shape parameter epsilon from the k-th nearest-neighbor
    spacing of the center set.
    """
    centers = np.asarray(pts, dtype=float)
    n = centers.shape[0]
    if n <= 1:
        return 1.0

    k = min(max(2, int(neighbor_k) + 1), n)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=k)
    dists = np.asarray(dists, dtype=float)
    if dists.ndim == 1:
        dists = dists[:, None]
    dk = dists[:, -1]
    positive = dk[dk > 0.0]
    ref_dist = float(np.median(positive)) if positive.size > 0 else 1.0
    ref_dist = max(ref_dist, 1e-12)
    return max(float(shape_scale) / ref_dist, 1e-12)



def greedy_farthest_subset_from_scores(
    pts: np.ndarray,
    scores: np.ndarray,
    n_select: int,
    min_separation: float,
) -> np.ndarray:
    """
    Greedy top-score selection with a minimum separation constraint.

    Parameters
    ----------
    pts : (M,3) ndarray
        Candidate points.
    scores : (M,) ndarray
        Scores, larger is better.
    n_select : int
        Number of points to select.
    min_separation : float
        Minimum Euclidean spacing between selected points.

    Returns
    -------
    selected : (K,) ndarray of int
        Selected indices into pts.
    """
    order = np.argsort(scores)[::-1]
    selected = []
    selected_pts = []

    for idx in order:
        x = pts[idx]
        if not selected_pts:
            selected.append(idx)
            selected_pts.append(x)
        else:
            d = np.linalg.norm(np.asarray(selected_pts) - x[None, :], axis=1)
            if np.all(d >= min_separation):
                selected.append(idx)
                selected_pts.append(x)

        if len(selected) >= n_select:
            break

    return np.asarray(selected, dtype=int)


def structured_initial_subset_indices(
    points: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """
    Deterministic sparse subset using voxelized first-hit selection.

    Kept local to this module so experimental refiners can downsample large
    structured point sets without importing helpers from script-like modules.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    target_size = max(1, min(int(target_size), n))
    if target_size >= n:
        return np.arange(n, dtype=np.int64)

    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    lengths = np.maximum(bbox_max - bbox_min, 1e-12)
    cell_volume = float(np.prod(lengths) / target_size)
    h = max(cell_volume ** (1.0 / 3.0), 1e-12)

    selected = np.zeros((0,), dtype=np.int64)
    for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        cell_size = max(h * scale, 1e-12)
        vox = np.floor((pts - bbox_min[None, :]) / cell_size).astype(np.int64)
        _, keep = np.unique(vox, axis=0, return_index=True)
        keep = np.sort(keep)
        selected = keep
        if keep.size >= target_size:
            selected = keep[:target_size]
            break

    if selected.size < target_size:
        remaining_mask = np.ones((n,), dtype=bool)
        remaining_mask[selected] = False
        remaining = np.where(remaining_mask)[0]
        need = target_size - selected.size
        if need > 0 and remaining.size > 0:
            step = max(1, remaining.size // need)
            extras = remaining[::step][:need]
            selected = np.concatenate([selected, extras])

    return np.asarray(np.unique(selected[:target_size]), dtype=np.int64)


def top_k_sorted_indices(scores: np.ndarray, k: int) -> np.ndarray:
    vals = np.asarray(scores, dtype=float).reshape(-1)
    if vals.size == 0 or k <= 0:
        return np.zeros((0,), dtype=int)
    take = min(int(k), int(vals.size))
    if take >= vals.size:
        return np.argsort(vals)[::-1].astype(int, copy=False)
    idx = np.argpartition(vals, -take)[-take:]
    return idx[np.argsort(vals[idx])[::-1]].astype(int, copy=False)


def surface_proximity_weights(sdf_values: np.ndarray, tau: float) -> np.ndarray:
    abs_sdf = np.abs(np.asarray(sdf_values, dtype=float).reshape(-1))
    tau_eff = max(float(tau), 1e-12)
    return np.exp(-abs_sdf / tau_eff)


def ordered_unique_indices(indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    _, first = np.unique(idx, return_index=True)
    return idx[np.sort(first)].astype(np.int64, copy=False)


def hash_index_array(indices: np.ndarray) -> Tuple[int, str]:
    idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    digest = hashlib.blake2b(idx.view(np.uint8), digest_size=16).hexdigest()
    return (int(idx.size), digest)


class _DenseLinearFactor:
    def __init__(self, matrix: np.ndarray):
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("matrix must be square")
        self.n = int(arr.shape[0])
        self._mode = "empty"
        self._factor = None
        if self.n == 0:
            return
        try:
            self._factor = linalg.cho_factor(arr, lower=True, check_finite=False)
            self._mode = "cho"
        except linalg.LinAlgError:
            self._factor = linalg.lu_factor(arr, check_finite=False)
            self._mode = "lu"

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        arr = np.asarray(rhs, dtype=float)
        if self.n == 0:
            return np.zeros_like(arr, dtype=float)
        if self._mode == "cho":
            return np.asarray(
                linalg.cho_solve(self._factor, arr, check_finite=False),
                dtype=float,
            )
        return np.asarray(
            linalg.lu_solve(self._factor, arr, check_finite=False),
            dtype=float,
        )

    __call__ = solve


class _SparseLUFactor:
    def __init__(self, matrix: sparse.csc_matrix):
        mat = matrix.tocsc()
        self.n = int(mat.shape[0])
        self._factor = sparse_linalg.splu(mat)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        arr = np.asarray(rhs, dtype=float)
        if arr.ndim == 1:
            return np.asarray(self._factor.solve(arr), dtype=float)
        cols = [self._factor.solve(arr[:, i]) for i in range(arr.shape[1])]
        return np.column_stack(cols).astype(float, copy=False)

    __call__ = solve


class _CholmodFactor:
    def __init__(self, factor: Any, n: int):
        self._factor = factor
        self.n = int(n)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(self._factor(np.asarray(rhs, dtype=float)), dtype=float)

    __call__ = solve


class _BlockSchurFactor:
    def __init__(
        self,
        base_factor: Any,
        K_on: sparse.csc_matrix,
        K_no: sparse.csr_matrix,
        schur_factor: Any,
    ):
        self.base_factor = base_factor
        self.K_on = K_on.tocsc()
        self.K_no = K_no.tocsr()
        self.schur_factor = schur_factor
        self.n_old = int(base_factor.n)
        self.n_new = int(schur_factor.n)
        self.n = self.n_old + self.n_new

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        arr = np.asarray(rhs, dtype=float)
        is_vector = arr.ndim == 1
        if is_vector:
            arr = arr[:, None]

        b_old = arr[:self.n_old]
        b_new = arr[self.n_old :]
        z_old = self.base_factor.solve(b_old)
        rhs_schur = b_new - self.K_no @ z_old
        x_new = self.schur_factor.solve(rhs_schur)
        correction = self.base_factor.solve(self.K_on @ x_new)
        x_old = z_old - correction
        out = np.vstack((x_old, x_new))
        return out[:, 0] if is_vector else out

    __call__ = solve


def factorize_spd(matrix: sparse.spmatrix | np.ndarray) -> Any:
    if sparse.issparse(matrix):
        mat_csc = matrix.tocsc()
        if _cholmod_cholesky is not None:
            try:
                return _CholmodFactor(_cholmod_cholesky(mat_csc), mat_csc.shape[0])
            except Exception:
                pass
        try:
            return _SparseLUFactor(mat_csc)
        except Exception:
            return _DenseLinearFactor(mat_csc.toarray())
    return _DenseLinearFactor(np.asarray(matrix, dtype=float))


def _timed_factorize_spd(
    matrix: sparse.spmatrix | np.ndarray,
    timings: Optional[Dict[str, float]] = None,
    key: str = "time_factorization",
) -> Any:
    t0 = time.perf_counter()
    factor = factorize_spd(matrix)
    dt = time.perf_counter() - t0
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + dt
        timings["time_factorization_total"] = timings.get("time_factorization_total", 0.0) + dt
    return factor


def _timed_factor_solve(
    factor: Any,
    rhs: np.ndarray,
    timings: Optional[Dict[str, float]] = None,
    key: str = "time_solve",
) -> np.ndarray:
    t0 = time.perf_counter()
    out = np.asarray(factor(rhs), dtype=float)
    dt = time.perf_counter() - t0
    if timings is not None:
        timings[key] = timings.get(key, 0.0) + dt
        timings["time_solve_total"] = timings.get("time_solve_total", 0.0) + dt
    return out


# ============================================================
# Main refiner
# ============================================================

@dataclass
class DWRRefinementResult:
    iteration: int
    train_indices: np.ndarray
    check_indices: np.ndarray
    rho_local: np.ndarray
    rho_global: float
    residual: np.ndarray
    weights: np.ndarray
    lambda_vec: np.ndarray
    psi: np.ndarray
    eta: np.ndarray
    selected_check_indices: np.ndarray
    selected_global_indices: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass
class _IncrementalLocalState:
    train_indices: np.ndarray
    X: np.ndarray
    y_residual: np.ndarray
    rho_local: np.ndarray
    rho_global: float
    tree: Optional[cKDTree]
    K: Optional[sparse.csc_matrix]
    factor: Any
    coeffs: Optional[np.ndarray]
    n_rebuilds: int
    n_updates_since_rebuild: int
    model_version: int
    last_update_mode: str = "full"
    last_rebuild_reason: str = "initial"


class HybridWendlandC2DWRRefinerOptimized:
    """
    Optimized hybrid local/global Wendland refiner.

    This class is intentionally separate from `HybridWendlandC2DWRRefiner` so
    behavior can be compared side-by-side while iterating on performance.

    Immediate improvements implemented here
    ---------------------------------------
    1) `refine_once()` avoids a second full refit by default.
    2) When `freeze_global=True`, global predictions are cached on the static
       master point set and reused by index lookup.
    3) Matrix assembly uses the lower-overhead `assemble_wendland_matrix()`
       path updated above.
    """

    def __init__(
        self,
        all_points: np.ndarray,
        all_sdf: np.ndarray,
        initial_local_train_indices: np.ndarray,
        initial_global_train_indices: np.ndarray,
        far_field_points: Optional[np.ndarray] = None,
        far_field_sdf: Optional[np.ndarray] = None,
        far_field_box_min: Optional[np.ndarray] = None,
        far_field_box_max: Optional[np.ndarray] = None,
        oml_point_indices: Optional[np.ndarray] = None,
        local_candidate_mask: Optional[np.ndarray] = None,
        global_candidate_mask: Optional[np.ndarray] = None,
        point_metadata: Optional[Dict[str, np.ndarray]] = None,
        neighbor_k: int = 24,
        radius_scale: float = 1.5,
        radius_quantile: float = 0.90,
        radius_inflate: float = 1.10,
        weight_power: float = 1.0,
        weight_eps: float = 1e-3,
        jitter: float = 1e-10,
        shortlist_size: Optional[int] = None,
        shortlist_fraction: float = 0.1,
        assembly_chunk_size: int = 4096,
        global_max_centers: Optional[int] = 5000,
        global_radius_scale: float = 5.0,
        global_radius_quantile: float = 0.90,
        global_radius_inflate: float = 1.10,
        global_ridge: float = 1e-8,
        far_field_neighbor_k: int = 12,
        far_field_radius_scale: float = 1.75,
        far_field_radius_quantile: float = 0.90,
        far_field_radius_inflate: float = 1.10,
        far_field_ridge: float = 1e-10,
        far_field_eval_chunk_size: int = 2048,
        far_field_blend_inner_distance: Optional[float] = None,
        far_field_blend_outer_distance: Optional[float] = None,
        local_error_weight: float = 1.0,
        freeze_global: bool = True,
        enable_adjoint: bool = True,
        shortlist_mode: str = "weight",
        shortlist_dist_gamma: float = 1.0,
        shortlist_band_zero_tol: float = 5e-3,
        shortlist_band_pos_tol: float = 5e-2,
        shortlist_band_neg_tol: float = 5e-2,
        shortlist_band_weights: Optional[Tuple[float, float, float, float]] = None,
        shortlist_score_clip: float = 50.0,
        shortlist_dist_beta: float = 0.35,
        shortlist_feature_alpha: float = 1.0,
        shortlist_feature_fraction: float = 0.25,
        shortlist_feature_tau: float = 5e-2,
        selection_score_mode: str = "eta",
        selection_feature_weight: float = 0.35,
        selection_curvature_weight: float = 0.25,
        selection_sign_weight: float = 0.5,
        selection_surface_tau: float = 5e-2,
        shortlist_diagnostics: bool = False,
    ):
        self.all_points = np.asarray(all_points, dtype=float)
        self.all_sdf = np.asarray(all_sdf, dtype=float)
        if (far_field_points is None) != (far_field_sdf is None):
            raise ValueError("far_field_points and far_field_sdf must be provided together")

        if far_field_points is None:
            self.far_field_points = np.zeros((0, 3), dtype=float)
            self.far_field_sdf = np.zeros((0,), dtype=float)
        else:
            self.far_field_points = np.asarray(far_field_points, dtype=float)
            self.far_field_sdf = np.asarray(far_field_sdf, dtype=float).reshape(-1)

        assert self.all_points.ndim == 2 and self.all_points.shape[1] == 3
        assert self.all_sdf.ndim == 1 and self.all_sdf.shape[0] == self.all_points.shape[0]
        assert self.far_field_points.ndim == 2 and self.far_field_points.shape[1] == 3
        assert self.far_field_sdf.ndim == 1 and self.far_field_sdf.shape[0] == self.far_field_points.shape[0]

        if self.all_points.shape[0] > 0:
            inferred_box_min = np.min(self.all_points, axis=0)
            inferred_box_max = np.max(self.all_points, axis=0)
        elif self.far_field_points.shape[0] > 0:
            inferred_box_min = np.min(self.far_field_points, axis=0)
            inferred_box_max = np.max(self.far_field_points, axis=0)
        else:
            inferred_box_min = np.zeros((3,), dtype=float)
            inferred_box_max = np.ones((3,), dtype=float)

        if far_field_box_min is None:
            self.far_field_box_min = inferred_box_min.astype(float, copy=True)
        else:
            self.far_field_box_min = np.asarray(far_field_box_min, dtype=float).reshape(3)
        if far_field_box_max is None:
            self.far_field_box_max = inferred_box_max.astype(float, copy=True)
        else:
            self.far_field_box_max = np.asarray(far_field_box_max, dtype=float).reshape(3)
        self.far_field_box_max = np.maximum(self.far_field_box_max, self.far_field_box_min + 1e-12)
        self.far_field_center = 0.5 * (self.far_field_box_min + self.far_field_box_max)
        self.far_field_reference_radius = max(
            0.5 * float(np.linalg.norm(self.far_field_box_max - self.far_field_box_min)),
            1e-12,
        )

        self.n_total = self.all_points.shape[0]
        if oml_point_indices is None:
            oml_point_indices = np.flatnonzero(np.isclose(self.all_sdf, 0.0))
        self.oml_point_indices = np.asarray(oml_point_indices, dtype=int)

        local_train_mask = np.zeros(self.n_total, dtype=bool)
        global_train_mask = np.zeros(self.n_total, dtype=bool)
        local_train_mask[np.asarray(initial_local_train_indices, dtype=int)] = True
        global_train_mask[np.asarray(initial_global_train_indices, dtype=int)] = True

        if local_candidate_mask is None:
            local_candidate_mask = np.ones(self.n_total, dtype=bool)
        if global_candidate_mask is None:
            global_candidate_mask = np.ones(self.n_total, dtype=bool)

        self.local_candidate_mask = np.asarray(local_candidate_mask, dtype=bool)
        self.global_candidate_mask = np.asarray(global_candidate_mask, dtype=bool)
        self.local_train_mask = local_train_mask
        self.global_train_mask = global_train_mask
        self._init_point_metadata(point_metadata)

        self.neighbor_k = neighbor_k
        self.radius_scale = radius_scale
        self.radius_quantile = radius_quantile
        self.radius_inflate = radius_inflate
        self.weight_power = weight_power
        self.weight_eps = weight_eps
        self.jitter = jitter
        self.shortlist_size = shortlist_size
        self.shortlist_fraction = shortlist_fraction
        self.assembly_chunk_size = max(1, int(assembly_chunk_size))
        self.global_max_centers = global_max_centers
        self.global_radius_scale = float(global_radius_scale)
        self.global_radius_quantile = float(global_radius_quantile)
        self.global_radius_inflate = float(global_radius_inflate)
        self.global_ridge = float(global_ridge)
        self.far_field_neighbor_k = int(far_field_neighbor_k)
        self.far_field_radius_scale = float(far_field_radius_scale)
        self.far_field_radius_quantile = float(far_field_radius_quantile)
        self.far_field_radius_inflate = float(far_field_radius_inflate)
        self.far_field_ridge = float(far_field_ridge)
        self.far_field_eval_chunk_size = max(1, int(far_field_eval_chunk_size))
        self.far_field_blend_inner_distance = (
            0.0 if far_field_blend_inner_distance is None else max(float(far_field_blend_inner_distance), 0.0)
        )
        self.far_field_blend_outer_distance = (
            0.0 if far_field_blend_outer_distance is None else max(float(far_field_blend_outer_distance), 0.0)
        )
        self.local_error_weight = float(local_error_weight)
        self.freeze_global = bool(freeze_global)
        self.enable_adjoint = bool(enable_adjoint)

        _valid_shortlist_modes = {"weight", "dist_x_weight", "banded_weight", "geometry_banded_weight"}
        if shortlist_mode not in _valid_shortlist_modes:
            raise ValueError(
                f"shortlist_mode={shortlist_mode!r} is not recognized; "
                f"choose one of {sorted(_valid_shortlist_modes)}"
            )
        self.shortlist_mode = shortlist_mode
        self.shortlist_dist_gamma = float(shortlist_dist_gamma)
        self.shortlist_band_zero_tol = float(shortlist_band_zero_tol)
        self.shortlist_band_pos_tol = float(shortlist_band_pos_tol)
        self.shortlist_band_neg_tol = float(shortlist_band_neg_tol)
        if shortlist_band_weights is None:
            shortlist_band_weights = (0.45, 0.20, 0.20, 0.15)
        if len(shortlist_band_weights) != 4:
            raise ValueError("shortlist_band_weights must contain four entries")
        band_weights = np.asarray(shortlist_band_weights, dtype=float)
        band_weights = np.maximum(band_weights, 0.0)
        band_weight_sum = float(np.sum(band_weights))
        if band_weight_sum <= 0.0:
            raise ValueError("shortlist_band_weights must have positive sum")
        self.shortlist_band_weights = tuple((band_weights / band_weight_sum).tolist())
        self.shortlist_score_clip = float(shortlist_score_clip)
        self.shortlist_dist_beta = float(shortlist_dist_beta)
        self.shortlist_feature_alpha = float(shortlist_feature_alpha)
        self.shortlist_feature_fraction = float(np.clip(shortlist_feature_fraction, 0.0, 1.0))
        self.shortlist_feature_tau = float(shortlist_feature_tau)
        valid_selection_modes = {"eta", "dwr_plus_geometry"}
        if selection_score_mode not in valid_selection_modes:
            raise ValueError(
                f"selection_score_mode={selection_score_mode!r} is not recognized; "
                f"choose one of {sorted(valid_selection_modes)}"
            )
        self.selection_score_mode = selection_score_mode
        self.selection_feature_weight = float(selection_feature_weight)
        self.selection_curvature_weight = float(selection_curvature_weight)
        self.selection_sign_weight = float(selection_sign_weight)
        self.selection_surface_tau = float(selection_surface_tau)
        # Temporary diagnostics to help understand shortlist/selection behavior.
        # When True, additional counts per SDF band are added to the info/diagnostics
        # dictionaries. This is intended to be removed after debugging.
        self.shortlist_diagnostics = bool(shortlist_diagnostics)

        self._last_fit_info = None
        self._far_field_shape_parameter_cached: Optional[float] = None
        self._far_field_blend_inner_radius_cached: Optional[float] = None
        self._far_field_blend_outer_radius_cached: Optional[float] = None
        self._cached_far_field_factor = None
        self._cached_far_field_tree: Optional[cKDTree] = None
        self._cached_c_far_field: Optional[np.ndarray] = None
        self._cached_far_field_values: Optional[np.ndarray] = None
        self._cached_far_field_values_mask: Optional[np.ndarray] = None
        self._far_field_model_version: int = 0
        self._local_tr_tree: Optional[cKDTree] = None
        self._local_rho_global: Optional[float] = None
        self._global_tr_tree: Optional[cKDTree] = None
        self._global_rho: Optional[float] = None
        self._cached_global_factor = None
        self._cached_c_global: Optional[np.ndarray] = None
        self._cached_global_train_key: Optional[bytes] = None
        self._cached_global_values: Optional[np.ndarray] = None
        self._cached_global_values_mask: Optional[np.ndarray] = None
        self._local_model_version: int = 0
        self._global_model_version: int = 0
        self._prediction_operator_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _make_points_cache_key(self, points: np.ndarray) -> Tuple[Any, ...]:
        pts = np.ascontiguousarray(np.asarray(points, dtype=float))
        digest = hashlib.blake2b(pts.view(np.uint8), digest_size=16).hexdigest()
        return (pts.shape, pts.dtype.str, digest)

    def _prediction_cache_key(
        self,
        branch: str,
        points: np.ndarray,
        rho: float,
        n_src: int,
        model_version: int,
    ) -> Tuple[Any, ...]:
        return (
            branch,
            self._make_points_cache_key(points),
            int(n_src),
            float(rho),
            int(model_version),
        )

    def _get_or_build_prediction_operator(
        self,
        branch: str,
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        rho: float,
        src_tree: Optional[cKDTree],
        model_version: int,
    ) -> sparse.csr_matrix:
        if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
            return sparse.csr_matrix((tgt_pts.shape[0], src_pts.shape[0]), dtype=float)

        cache_key = self._prediction_cache_key(
            branch=branch,
            points=tgt_pts,
            rho=rho,
            n_src=src_pts.shape[0],
            model_version=model_version,
        )
        cached = self._prediction_operator_cache.get(cache_key)
        if cached is not None:
            return cached["operator"]

        op = assemble_wendland_matrix_from_sparse_distance(
            src_pts,
            tgt_pts,
            rho,
            src_tree=src_tree,
        )
        self._prediction_operator_cache[cache_key] = {"operator": op}
        return op

    def _init_point_metadata(self, point_metadata: Optional[Dict[str, np.ndarray]]) -> None:
        def _vector(
            name: str,
            dtype: Any,
            default_value: float | int | bool = 0,
        ) -> np.ndarray:
            if point_metadata is None or name not in point_metadata:
                return np.full((self.n_total,), default_value, dtype=dtype)
            arr = np.asarray(point_metadata[name], dtype=dtype).reshape(-1)
            if arr.shape[0] != self.n_total:
                raise ValueError(
                    f"point_metadata[{name!r}] must have length {self.n_total}, "
                    f"got {arr.shape[0]}"
                )
            return arr

        self.point_feature_strength = _vector("feature_strength", float, 0.0)
        self.point_curvature = _vector("curvature", float, 0.0)
        self.point_curvature_score = _vector("curvature_score", float, 0.0)
        self.point_normal_variation = _vector("normal_variation", float, 0.0)
        self.point_curvature_variation = _vector("curvature_variation", float, 0.0)
        self.point_sharpness = _vector("sharpness", float, 0.0)
        self.point_geometry_valid = _vector("valid", bool, False)
        self.point_source_kind = _vector("source_kind", np.int8, -1)
        self.point_surface_seed_index = _vector("surface_seed_index", np.int64, -1)
        self.has_geometry_metadata = bool(np.any(self.point_geometry_valid))

    def _geometry_feature_for_indices(self, point_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(point_indices, dtype=int)
        if idx.size == 0:
            return np.zeros((0,), dtype=float)
        feature = np.asarray(self.point_feature_strength[idx], dtype=float).reshape(-1)
        valid = np.asarray(self.point_geometry_valid[idx], dtype=bool).reshape(-1)
        feature[~valid] = 0.0
        return feature

    def _curvature_score_for_indices(self, point_indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(point_indices, dtype=int)
        if idx.size == 0:
            return np.zeros((0,), dtype=float)
        curvature_score = np.asarray(self.point_curvature_score[idx], dtype=float).reshape(-1)
        valid = np.asarray(self.point_geometry_valid[idx], dtype=bool).reshape(-1)
        curvature_score[~valid] = 0.0
        return curvature_score

    def _compute_local_selection_scores(
        self,
        point_indices: np.ndarray,
        sdf_values: np.ndarray,
        pred_values: np.ndarray,
        eta_base: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        n = int(np.asarray(point_indices, dtype=int).size)
        zeros = np.zeros((n,), dtype=float)
        if n == 0:
            return zeros, {
                "feature_bonus": zeros,
                "curvature_bonus": zeros,
                "sign_bonus": zeros,
                "surface_proximity": zeros,
            }

        if self.selection_score_mode != "dwr_plus_geometry" or not self.has_geometry_metadata:
            return np.asarray(eta_base, dtype=float).reshape(-1), {
                "feature_bonus": zeros,
                "curvature_bonus": zeros,
                "sign_bonus": zeros,
                "surface_proximity": zeros,
            }

        sdf_vals = np.asarray(sdf_values, dtype=float).reshape(-1)
        pred_vals = np.asarray(pred_values, dtype=float).reshape(-1)
        eta_vals = np.asarray(eta_base, dtype=float).reshape(-1)
        surface_proximity = surface_proximity_weights(sdf_vals, self.selection_surface_tau)
        feature_bonus = self.selection_feature_weight * (
            surface_proximity * self._geometry_feature_for_indices(point_indices)
        )
        curvature_bonus = self.selection_curvature_weight * (
            surface_proximity * self._curvature_score_for_indices(point_indices)
        )

        sign_bonus = np.zeros((n,), dtype=float)
        tau0 = max(float(self.shortlist_band_zero_tol), 1e-12)
        off_surface = np.abs(sdf_vals) > float(self.shortlist_band_zero_tol)
        sign_bonus[off_surface] = (
            pred_vals[off_surface] * sdf_vals[off_surface] < 0.0
        ).astype(float)
        sign_bonus[~off_surface] = np.minimum(np.abs(pred_vals[~off_surface]) / tau0, 1.0)
        sign_bonus = self.selection_sign_weight * sign_bonus

        # multiplier =   sign_bonus + feature_bonus # + curvature_bonus +
        multiplier = 1.0 + feature_bonus + curvature_bonus + sign_bonus
        selection_scores = eta_vals * multiplier
        # selection_scores = eta_vals + multiplier
        # selection_scores = multiplier
        return selection_scores, {
            "feature_bonus": feature_bonus,
            "curvature_bonus": curvature_bonus,
            "sign_bonus": sign_bonus,
            "surface_proximity": surface_proximity,
        }

    def _compute_local_oml_max_abs_error(
        self,
        X_global: np.ndarray,
        c_global: np.ndarray,
        global_rho: float,
        global_tr_tree: Optional[cKDTree],
        X_local: np.ndarray,
        c_local: np.ndarray,
        rho_global: float,
        local_tr_tree: Optional[cKDTree],
    ) -> float:
        oml_idx = self.oml_point_indices
        if oml_idx.size == 0:
            return 0.0

        pred = np.zeros((oml_idx.size,), dtype=float)
        oml_pts = self.all_points[oml_idx]

        if X_global.shape[0] > 0:
            pred += self._evaluate_global_model_on_indices(
                oml_idx,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )

        if X_local.shape[0] > 0:
            A_local_oml = assemble_wendland_matrix(
                X_local,
                oml_pts,
                rho_global,
                src_tree=local_tr_tree,
                chunk_size=self.assembly_chunk_size,
            )
            pred += np.asarray(A_local_oml @ c_local, dtype=float).reshape(-1)

        return float(np.max(np.abs(pred - self.all_sdf[oml_idx])))

    def _current_indices(self) -> Dict[str, np.ndarray]:
        local_train_idx = np.where(self.local_train_mask)[0]
        global_train_idx = np.where(self.global_train_mask)[0]

        local_check_mask = self.local_candidate_mask & (~self.local_train_mask)
        if self.freeze_global:
            global_check_idx = np.array([], dtype=np.int64)
        else:
            global_check_mask = self.global_candidate_mask & (~self.global_train_mask)
            global_check_idx = np.where(global_check_mask)[0]

        return {
            "local_train_idx": local_train_idx,
            "global_train_idx": global_train_idx,
            "local_check_idx": np.where(local_check_mask)[0],
            "global_check_idx": global_check_idx,
        }

    def _compute_global_support_radius(self, Xg: np.ndarray) -> float:
        if Xg.shape[0] <= 1:
            return 1.0
        rho_local = compute_local_support_radii(
            Xg,
            neighbor_k=self.neighbor_k,
            radius_scale=self.global_radius_scale,
        )
        return robust_global_support_radius(
            rho_local,
            quantile=self.global_radius_quantile,
            inflate=self.global_radius_inflate,
        )

    def _has_far_field_stage(self) -> bool:
        return self.far_field_points.shape[0] > 0

    def _compute_far_field_shape_parameter(self, X_far_field: np.ndarray) -> float:
        if X_far_field.shape[0] <= 1:
            return 1.0
        rho_local = compute_local_support_radii(
            X_far_field,
            neighbor_k=self.far_field_neighbor_k,
            radius_scale=self.far_field_radius_scale,
        )
        return robust_global_support_radius(
            rho_local,
            quantile=self.far_field_radius_quantile,
            inflate=self.far_field_radius_inflate,
        )

    def _compute_far_field_blend_radii(self, X_far_field: np.ndarray) -> Tuple[float, float]:
        if X_far_field.shape[0] == 0:
            return 0.0, 0.0
        return float(self.far_field_blend_inner_distance), float(self.far_field_blend_outer_distance)

    def _signed_far_field_box_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] == 0:
            return np.zeros((0,), dtype=float)

        box_min = self.far_field_box_min[None, :]
        box_max = self.far_field_box_max[None, :]
        below = box_min - pts
        above = pts - box_max
        outside = np.maximum(np.maximum(below, above), 0.0)
        outside_dist = np.max(outside, axis=1)

        inside_margin = np.minimum(pts - box_min, box_max - pts)
        inside_dist = np.min(inside_margin, axis=1)
        inside_mask = np.all((pts >= box_min) & (pts <= box_max), axis=1)

        signed = outside_dist
        signed[inside_mask] = -inside_dist[inside_mask]
        return signed

    def _evaluate_far_field_blend_weights(
        self,
        points: np.ndarray,
        inner_radius: float,
        outer_radius: float,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] == 0:
            return np.zeros((0,), dtype=float)
        signed_distance = self._signed_far_field_box_distance(pts)
        transition_width = float(inner_radius) + float(outer_radius)
        if transition_width <= 1e-12:
            return (signed_distance >= 0.0).astype(float)

        t = (signed_distance + float(inner_radius)) / transition_width
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _fit_far_field_stage(self) -> Dict[str, Any]:
        X_far_field = self.far_field_points
        y_far_field = self.far_field_sdf
        n_far = X_far_field.shape[0]
        if n_far == 0:
            return {
                "X_far_field": X_far_field,
                "y_far_field": y_far_field,
                "K_far_field": sparse.csc_matrix((0, 0), dtype=float),
                "far_field_factor": None,
                "c_far_field": np.zeros((0,), dtype=float),
                "far_field_center": self.far_field_center.copy(),
                "far_field_reference_radius": float(self.far_field_reference_radius),
                "far_field_shape_parameter": 1.0,
                "far_field_length_scale": 1.0,
                "far_field_blend_inner_radius": 0.0,
                "far_field_blend_outer_radius": 0.0,
                "far_field_box_min": self.far_field_box_min.copy(),
                "far_field_box_max": self.far_field_box_max.copy(),
                "far_field_rho": 1.0,
                "far_field_blend_inner_distance": 0.0,
                "far_field_blend_outer_distance": 0.0,
            }

        if (
            self._cached_far_field_factor is not None
            and self._cached_far_field_tree is not None
            and self._cached_c_far_field is not None
            and self._far_field_shape_parameter_cached is not None
            and self._far_field_blend_inner_radius_cached is not None
            and self._far_field_blend_outer_radius_cached is not None
        ):
            return {
                "X_far_field": X_far_field,
                "y_far_field": y_far_field,
                "K_far_field": sparse.csc_matrix((n_far, n_far), dtype=float),
                "far_field_factor": self._cached_far_field_factor,
                "c_far_field": self._cached_c_far_field,
                "far_field_center": self.far_field_center.copy(),
                "far_field_reference_radius": float(self.far_field_reference_radius),
                "far_field_shape_parameter": self._far_field_shape_parameter_cached,
                "far_field_length_scale": self._far_field_shape_parameter_cached,
                "far_field_blend_inner_radius": self._far_field_blend_inner_radius_cached,
                "far_field_blend_outer_radius": self._far_field_blend_outer_radius_cached,
                "far_field_box_min": self.far_field_box_min.copy(),
                "far_field_box_max": self.far_field_box_max.copy(),
                "far_field_rho": self._far_field_shape_parameter_cached,
                "far_field_blend_inner_distance": self._far_field_blend_inner_radius_cached,
                "far_field_blend_outer_distance": self._far_field_blend_outer_radius_cached,
            }

        far_field_rho = self._compute_far_field_shape_parameter(X_far_field)
        blend_inner_radius, blend_outer_radius = self._compute_far_field_blend_radii(X_far_field)
        far_field_tree = cKDTree(X_far_field)
        K_far_field = assemble_wendland_matrix(
            X_far_field,
            X_far_field,
            far_field_rho,
            src_tree=far_field_tree,
            chunk_size=self.assembly_chunk_size,
        ).tocsc()
        K_far_field = K_far_field + self.far_field_ridge * sparse.eye(K_far_field.shape[0], format="csc")
        far_field_factor = factorize_spd(K_far_field)
        c_far_field = np.asarray(far_field_factor(y_far_field), dtype=float).reshape(-1)

        self._far_field_shape_parameter_cached = far_field_rho
        self._far_field_blend_inner_radius_cached = blend_inner_radius
        self._far_field_blend_outer_radius_cached = blend_outer_radius
        self._cached_far_field_factor = far_field_factor
        self._cached_far_field_tree = far_field_tree
        self._cached_c_far_field = c_far_field
        self._cached_far_field_values = np.zeros((self.n_total,), dtype=float)
        self._cached_far_field_values_mask = np.zeros((self.n_total,), dtype=bool)

        return {
            "X_far_field": X_far_field,
            "y_far_field": y_far_field,
            "K_far_field": K_far_field,
            "far_field_factor": far_field_factor,
            "c_far_field": c_far_field,
            "far_field_center": self.far_field_center.copy(),
            "far_field_reference_radius": float(self.far_field_reference_radius),
            "far_field_shape_parameter": far_field_rho,
            "far_field_length_scale": far_field_rho,
            "far_field_blend_inner_radius": blend_inner_radius,
            "far_field_blend_outer_radius": blend_outer_radius,
            "far_field_box_min": self.far_field_box_min.copy(),
            "far_field_box_max": self.far_field_box_max.copy(),
            "far_field_rho": far_field_rho,
            "far_field_blend_inner_distance": blend_inner_radius,
            "far_field_blend_outer_distance": blend_outer_radius,
        }

    def _evaluate_far_field_values(
        self,
        points: np.ndarray,
        X_far_field: np.ndarray,
        c_far_field: np.ndarray,
        far_field_shape_parameter: float,
        far_field_blend_inner_radius: float,
        far_field_blend_outer_radius: float,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] == 0 or X_far_field.shape[0] == 0:
            return np.zeros((pts.shape[0],), dtype=float)
        raw = evaluate_rbf_matvec(
            X_far_field,
            pts,
            far_field_shape_parameter,
            c_far_field,
            src_tree=self._cached_far_field_tree,
        )
        weights = self._evaluate_far_field_blend_weights(
            pts,
            far_field_blend_inner_radius,
            far_field_blend_outer_radius,
        )
        return raw * weights

    def _evaluate_far_field_on_indices(
        self,
        point_indices: np.ndarray,
        X_far_field: np.ndarray,
        c_far_field: np.ndarray,
        far_field_shape_parameter: float,
        far_field_blend_inner_radius: float,
        far_field_blend_outer_radius: float,
    ) -> np.ndarray:
        idx = np.asarray(point_indices, dtype=int)
        if idx.size == 0 or X_far_field.shape[0] == 0:
            return np.zeros((idx.size,), dtype=float)

        if self.freeze_global and self._cached_far_field_values is not None:
            assert self._cached_far_field_values_mask is not None
            if np.all(self._cached_far_field_values_mask[idx]):
                return self._cached_far_field_values[idx].copy()

        pred = self._evaluate_far_field_values(
            self.all_points[idx],
            X_far_field,
            c_far_field,
            far_field_shape_parameter,
            far_field_blend_inner_radius,
            far_field_blend_outer_radius,
        )

        if self.freeze_global:
            if self._cached_far_field_values is None:
                self._cached_far_field_values = np.zeros((self.n_total,), dtype=float)
                self._cached_far_field_values_mask = np.zeros((self.n_total,), dtype=bool)
            self._cached_far_field_values[idx] = pred
            self._cached_far_field_values_mask[idx] = True
        return pred

    def _evaluate_background_model_on_indices(
        self,
        point_indices: np.ndarray,
        X_far_field: np.ndarray,
        c_far_field: np.ndarray,
        far_field_shape_parameter: float,
        far_field_blend_inner_radius: float,
        far_field_blend_outer_radius: float,
        X_global: np.ndarray,
        c_global: np.ndarray,
        global_rho: float,
        global_tr_tree: Optional[cKDTree],
    ) -> np.ndarray:
        idx = np.asarray(point_indices, dtype=int)
        pred = np.zeros((idx.size,), dtype=float)
        if X_far_field.shape[0] > 0:
            pred += self._evaluate_far_field_on_indices(
                idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
            )
        if X_global.shape[0] > 0:
            pred += self._evaluate_global_model_on_indices(
                idx,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )
        return pred

    def _compute_shortlist_indices(
        self,
        scores: np.ndarray,
        sdf_values: Optional[np.ndarray] = None,
        X_check: Optional[np.ndarray] = None,
        tr_tree: Optional[cKDTree] = None,
        point_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return indices of the top-*k* candidates to evaluate the surrogate on.

        Parameters
        ----------
        scores : np.ndarray, shape (n_check,)
            Base priority scores, typically ``w_i = 1/(|sdf_i|+eps)``.
        sdf_values : np.ndarray, shape (n_check,), optional
            Exact SDF values at the candidate points. Required when
            ``self.shortlist_mode`` is ``"banded_weight"`` or
            ``"geometry_banded_weight"``.
        X_check : np.ndarray, shape (n_check, 3), optional
            Spatial coordinates of the check candidates.  Required when
            ``self.shortlist_mode`` is ``"dist_x_weight"`` or
            ``"banded_weight"`` or ``"geometry_banded_weight"``.
        tr_tree : cKDTree, optional
            KD-tree of current training points used to compute the distance
            from each candidate to its nearest training point.  Required when
            ``self.shortlist_mode`` is ``"dist_x_weight"`` or
            ``"banded_weight"`` or ``"geometry_banded_weight"``.
        point_indices : np.ndarray, shape (n_check,), optional
            Indices of the candidates in ``self.all_points``. Required only for
            geometry-aware shortlist modes so the precomputed point metadata can
            be queried.
        """
        n = scores.size
        if n == 0:
            return np.array([], dtype=int)
        shortlist_n = self.shortlist_size
        if shortlist_n is None:
            shortlist_n = int(np.ceil(self.shortlist_fraction * n))
        shortlist_n = max(1, min(int(shortlist_n), n))
        if shortlist_n >= n:
            return np.arange(n, dtype=int)

        if self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            if sdf_values is None:
                raise ValueError(
                    "sdf_values are required for shortlist_mode in "
                    "{'banded_weight', 'geometry_banded_weight'}"
                )

            sdf_values = np.asarray(sdf_values, dtype=float).reshape(-1)
            if sdf_values.shape[0] != n:
                raise ValueError("sdf_values must have the same length as scores")

            if X_check is not None and tr_tree is not None:
                d_to_train, _ = tr_tree.query(X_check, k=1, workers=-1)
                d_to_train = np.maximum(np.asarray(d_to_train, dtype=float).reshape(-1), 0.0)
                d_ref = float(np.median(d_to_train)) if d_to_train.size > 0 else 1.0
                if not np.isfinite(d_ref) or d_ref <= 0.0:
                    positive = d_to_train[d_to_train > 0.0]
                    d_ref = float(np.median(positive)) if positive.size > 0 else 1.0
                coverage = 1.0 + self.shortlist_dist_beta * (
                    d_to_train / max(d_ref, 1e-12)
                ) ** self.shortlist_dist_gamma
            else:
                coverage = np.ones((n,), dtype=float)

            clipped_scores = np.minimum(np.asarray(scores, dtype=float), self.shortlist_score_clip)
            combined_scores = clipped_scores * coverage
            geometry_feature_scores = np.zeros((n,), dtype=float)
            use_geometry_shortlist = (
                self.shortlist_mode == "geometry_banded_weight"
                and point_indices is not None
                and self.has_geometry_metadata
            )
            if use_geometry_shortlist:
                geometry_feature_scores = (
                    surface_proximity_weights(sdf_values, self.shortlist_feature_tau)
                    * self._geometry_feature_for_indices(np.asarray(point_indices, dtype=int))
                )
                combined_scores = combined_scores * (
                    1.0 + self.shortlist_feature_alpha * geometry_feature_scores
                )

            tau0 = self.shortlist_band_zero_tol
            tau_pos = self.shortlist_band_pos_tol
            tau_neg = self.shortlist_band_neg_tol

            band_masks = [
                np.abs(sdf_values) <= tau0,
                (sdf_values > tau0) & (sdf_values <= tau_pos),
                (sdf_values < -tau0) & (sdf_values >= -tau_neg),
                sdf_values < -tau_neg,
            ]

            band_indices = [np.flatnonzero(mask) for mask in band_masks]
            band_order = [0, 1, 2, 3]
            quotas_float = np.asarray(self.shortlist_band_weights, dtype=float) * float(shortlist_n)
            quotas = np.floor(quotas_float).astype(int)
            remainder = int(shortlist_n - np.sum(quotas))
            if remainder > 0:
                frac_order = np.argsort(-(quotas_float - quotas))
                for idx in frac_order[:remainder]:
                    quotas[idx] += 1

            selected_parts = []
            selected_total = 0
            leftover = 0
            for band_id in band_order:
                idx_band = band_indices[band_id]
                target = quotas[band_id] + leftover
                if idx_band.size == 0 or target <= 0:
                    leftover = max(target, 0)
                    continue
                take = min(int(target), int(idx_band.size))
                chosen_parts = []

                if use_geometry_shortlist:
                    band_feature_scores = geometry_feature_scores[idx_band]
                    positive_feature = band_feature_scores > 0.0
                    feature_take = 0
                    if np.any(positive_feature) and self.shortlist_feature_fraction > 0.0:
                        feature_take = int(np.round(self.shortlist_feature_fraction * float(take)))
                        if feature_take == 0 and take > 0:
                            feature_take = 1
                        feature_take = min(int(take), int(np.count_nonzero(positive_feature)), int(feature_take))
                    if feature_take > 0:
                        feat_idx_local = top_k_sorted_indices(band_feature_scores, feature_take)
                        if feat_idx_local.size > 0:
                            chosen_parts.append(idx_band[feat_idx_local])

                already_chosen = (
                    np.unique(np.concatenate(chosen_parts)).astype(int, copy=False)
                    if chosen_parts
                    else np.zeros((0,), dtype=int)
                )
                remaining_band_mask = np.ones((idx_band.size,), dtype=bool)
                if already_chosen.size > 0:
                    remaining_band_mask[np.isin(idx_band, already_chosen)] = False
                remaining_band = idx_band[remaining_band_mask]
                remaining_take = int(take - already_chosen.size)
                if remaining_take > 0 and remaining_band.size > 0:
                    proxy_idx_local = top_k_sorted_indices(combined_scores[remaining_band], remaining_take)
                    if proxy_idx_local.size > 0:
                        chosen_parts.append(remaining_band[proxy_idx_local])

                chosen = (
                    np.unique(np.concatenate(chosen_parts)).astype(int, copy=False)
                    if chosen_parts
                    else np.zeros((0,), dtype=int)
                )
                if chosen.size > 0:
                    order = np.argsort(combined_scores[chosen])[::-1]
                    chosen = chosen[order]
                selected_parts.append(chosen)
                selected_total += chosen.size
                leftover = int(target - chosen.size)

            if selected_total < shortlist_n:
                already = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
                remaining_mask = np.ones((n,), dtype=bool)
                remaining_mask[already] = False
                remaining_idx = np.flatnonzero(remaining_mask)
                if remaining_idx.size > 0:
                    need = min(int(shortlist_n - selected_total), int(remaining_idx.size))
                    rem_short = top_k_sorted_indices(combined_scores[remaining_idx], need)
                    extra = remaining_idx[rem_short]
                    selected_parts.append(extra)

            shortlist = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
            if shortlist.size == 0:
                return shortlist
            shortlist_scores = combined_scores[shortlist]
            return shortlist[np.argsort(shortlist_scores)[::-1]].astype(int, copy=False)

        if (
            self.shortlist_mode == "dist_x_weight"
            and X_check is not None
            and tr_tree is not None
        ):
            # Combined proxy:  s_i = w_i * d_i^gamma
            # d_i = distance from candidate i to nearest current training point.
            # This boosts candidates that are both near the surface (high w_i)
            # AND far from existing training data (high d_i), so the shortlist
            # covers the full SDF spectrum rather than concentrating at the OML.
            d_to_train, _ = tr_tree.query(X_check, k=1, workers=-1)
            d_to_train = np.maximum(d_to_train, 0.0)
            combined_scores = scores * (d_to_train ** self.shortlist_dist_gamma)
        else:
            combined_scores = scores

        shortlist = top_k_sorted_indices(combined_scores, shortlist_n)
        return shortlist[np.argsort(combined_scores[shortlist])[::-1]].astype(int, copy=False)

    def _evaluate_global_model_on_indices(
        self,
        point_indices: np.ndarray,
        X_global: np.ndarray,
        c_global: np.ndarray,
        global_rho: float,
        global_tr_tree: Optional[cKDTree],
    ) -> np.ndarray:
        point_indices = np.asarray(point_indices, dtype=int)
        if point_indices.size == 0 or X_global.shape[0] == 0:
            return np.zeros((point_indices.size,), dtype=float)

        if self.freeze_global and self._cached_global_values is not None:
            assert self._cached_global_values_mask is not None
            if np.all(self._cached_global_values_mask[point_indices]):
                return self._cached_global_values[point_indices].copy()

        points = self.all_points[point_indices]
        A = assemble_wendland_matrix_from_sparse_distance(
            X_global,
            points,
            global_rho,
            src_tree=global_tr_tree,
        )
        pred = np.asarray(A @ c_global, dtype=float).reshape(-1)

        if self.freeze_global:
            if self._cached_global_values is None:
                self._cached_global_values = np.zeros((self.n_total,), dtype=float)
                self._cached_global_values_mask = np.zeros((self.n_total,), dtype=bool)
            self._cached_global_values[point_indices] = pred
            self._cached_global_values_mask[point_indices] = True

        return pred

    def _invalidate_fit_cache(self) -> None:
        self._last_fit_info = None

    def _invalidate_local_cache(self) -> None:
        self._local_tr_tree = None
        self._local_rho_global = None
        self._local_model_version += 1
        self._prediction_operator_cache = {
            k: v for k, v in self._prediction_operator_cache.items() if k[0] != "local"
        }
        self._invalidate_fit_cache()

    def _invalidate_global_cache(self) -> None:
        self._global_tr_tree = None
        self._global_rho = None
        self._cached_global_factor = None
        self._cached_c_global = None
        self._cached_global_train_key = None
        self._cached_global_values = None
        self._cached_global_values_mask = None
        self._global_model_version += 1
        self._prediction_operator_cache = {
            k: v for k, v in self._prediction_operator_cache.items() if k[0] != "global"
        }
        self._invalidate_fit_cache()

    def _fit_current_model(self) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        timings = {
            "time_fit_global": 0.0,
            "time_fit_local_full": 0.0,
            "time_shortlist": 0.0,
            "time_shortlist_prediction": 0.0,
            "time_adjoint": 0.0,
            "time_total_fit": 0.0,
            "time_factorization_total": 0.0,
            "time_solve_total": 0.0,
            "time_global_factorization": 0.0,
            "time_global_solve": 0.0,
            "time_local_factorization": 0.0,
            "time_local_solve": 0.0,
            "time_adjoint_solve": 0.0,
            "time_oml_surrogate_eval_total": 0.0,
            "time_oml_surrogate_eval_per_point": 0.0,
        }
        idx = self._current_indices()

        local_train_idx = idx["local_train_idx"]
        global_train_idx = idx["global_train_idx"]
        local_check_idx = idx["local_check_idx"]
        global_check_idx = idx["global_check_idx"]

        X_local = self.all_points[local_train_idx]
        y_local = self.all_sdf[local_train_idx]
        X_global = self.all_points[global_train_idx]
        y_global = self.all_sdf[global_train_idx]
        far_field_info = self._fit_far_field_stage()
        X_far_field = far_field_info["X_far_field"]
        y_far_field = far_field_info["y_far_field"]
        K_far_field = far_field_info["K_far_field"]
        far_field_factor = far_field_info["far_field_factor"]
        c_far_field = far_field_info["c_far_field"]
        far_field_center = far_field_info["far_field_center"]
        far_field_reference_radius = far_field_info["far_field_reference_radius"]
        far_field_shape_parameter = far_field_info["far_field_shape_parameter"]
        far_field_length_scale = far_field_info["far_field_length_scale"]
        far_field_blend_inner_radius = far_field_info["far_field_blend_inner_radius"]
        far_field_blend_outer_radius = far_field_info["far_field_blend_outer_radius"]
        far_field_box_min = far_field_info["far_field_box_min"]
        far_field_box_max = far_field_info["far_field_box_max"]
        far_field_rho = far_field_info["far_field_rho"]
        far_field_blend_inner_distance = far_field_info["far_field_blend_inner_distance"]
        far_field_blend_outer_distance = far_field_info["far_field_blend_outer_distance"]

        if self.global_max_centers is not None and X_global.shape[0] > int(self.global_max_centers):
            keep = structured_initial_subset_indices(X_global, int(self.global_max_centers))
            X_global = X_global[keep]
            y_global = y_global[keep]
            global_train_idx = global_train_idx[keep]

        global_train_key = global_train_idx.tobytes()
        global_cache_valid = (
            self._cached_global_factor is not None
            and self._cached_c_global is not None
            and self._cached_global_train_key == global_train_key
            and self._global_tr_tree is not None
            and self._global_rho is not None
        )

        if global_cache_valid:
            global_rho = self._global_rho
            global_tr_tree = self._global_tr_tree
            global_factor = self._cached_global_factor
            c_global = self._cached_c_global
            K_global = sparse.csc_matrix((X_global.shape[0], X_global.shape[0]), dtype=float)
        elif X_global.shape[0] > 0:
            fit_global_t0 = time.perf_counter()
            global_rho = self._compute_global_support_radius(X_global)
            global_tr_tree = cKDTree(X_global)
            global_background = self._evaluate_far_field_on_indices(
                global_train_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
            )
            y_global_residual = y_global - global_background
            K_global = assemble_wendland_matrix(
                X_global,
                X_global,
                global_rho,
                src_tree=global_tr_tree,
                chunk_size=self.assembly_chunk_size,
            ).tocsc()
            K_global = K_global + self.global_ridge * sparse.eye(K_global.shape[0], format="csc")
            global_factor = _timed_factorize_spd(
                K_global,
                timings=timings,
                key="time_global_factorization",
            )
            c_global = _timed_factor_solve(
                global_factor,
                y_global_residual,
                timings=timings,
                key="time_global_solve",
            )
            timings["time_fit_global"] += time.perf_counter() - fit_global_t0
            self._global_tr_tree = global_tr_tree
            self._global_rho = global_rho
            self._cached_global_factor = global_factor
            self._cached_c_global = c_global
            self._cached_global_train_key = global_train_key
            if self.freeze_global:
                self._cached_global_values = np.zeros((self.n_total,), dtype=float)
                self._cached_global_values_mask = np.zeros((self.n_total,), dtype=bool)
        else:
            global_rho = 1.0
            global_tr_tree = None
            global_factor = None
            c_global = np.zeros((0,), dtype=float)
            K_global = sparse.csc_matrix((0, 0), dtype=float)

        local_background = self._evaluate_background_model_on_indices(
            local_train_idx,
            X_far_field,
            c_far_field,
            far_field_shape_parameter,
            far_field_blend_inner_radius,
            far_field_blend_outer_radius,
            X_global,
            c_global,
            global_rho,
            global_tr_tree,
        )
        y_local_residual = y_local - local_background

        if self._local_tr_tree is None or self._local_rho_global is None:
            rho_local = compute_local_support_radii(
                X_local,
                neighbor_k=self.neighbor_k,
                radius_scale=self.radius_scale,
            ) if X_local.shape[0] > 0 else np.zeros((0,), dtype=float)
            rho_global = robust_global_support_radius(
                rho_local,
                quantile=self.radius_quantile,
                inflate=self.radius_inflate,
            ) if rho_local.size > 0 else 1.0
            local_tr_tree = cKDTree(X_local) if X_local.shape[0] > 0 else None
            self._local_tr_tree = local_tr_tree
            self._local_rho_global = rho_global
        else:
            rho_local = compute_local_support_radii(
                X_local,
                neighbor_k=self.neighbor_k,
                radius_scale=self.radius_scale,
            ) if X_local.shape[0] > 0 else np.zeros((0,), dtype=float)
            rho_global = self._local_rho_global
            local_tr_tree = self._local_tr_tree

        if X_local.shape[0] > 0:
            fit_local_t0 = time.perf_counter()
            K_local = assemble_wendland_matrix(
                X_local,
                X_local,
                rho_global,
                src_tree=local_tr_tree,
                chunk_size=self.assembly_chunk_size,
            ).tocsc()
            K_local = K_local + self.jitter * sparse.eye(K_local.shape[0], format="csc")
            local_factor = _timed_factorize_spd(
                K_local,
                timings=timings,
                key="time_local_factorization",
            )
            c_local = _timed_factor_solve(
                local_factor,
                y_local_residual,
                timings=timings,
                key="time_local_solve",
            )
            timings["time_fit_local_full"] += time.perf_counter() - fit_local_t0
        else:
            K_local = sparse.csc_matrix((0, 0), dtype=float)
            local_factor = None
            c_local = np.zeros((0,), dtype=float)

        local_weights = inverse_distance_weights(
            self.all_sdf[local_check_idx],
            power=self.weight_power,
            eps=self.weight_eps,
            clip_max=1e8,
            normalize=True,
        ) if local_check_idx.size > 0 else np.zeros((0,), dtype=float)
        global_weights = inverse_distance_weights(
            self.all_sdf[global_check_idx],
            power=self.weight_power,
            eps=self.weight_eps,
            clip_max=1e8,
            normalize=True,
        ) if global_check_idx.size > 0 else np.zeros((0,), dtype=float)

        # Shortlist: "weight" uses w_i only; "dist_x_weight" multiplies by
        # distance to nearest training point to spread evaluations across the
        # full SDF spectrum instead of concentrating at the OML.
        shortlist_t0 = time.perf_counter()
        X_local_check = self.all_points[local_check_idx]
        local_shortlist = self._compute_shortlist_indices(
            local_weights,
            sdf_values=self.all_sdf[local_check_idx],
            X_check=X_local_check if X_local_check.size > 0 else None,
            tr_tree=local_tr_tree,
            point_indices=local_check_idx,
        )
        X_global_check = self.all_points[global_check_idx]
        global_shortlist = self._compute_shortlist_indices(
            global_weights,
            sdf_values=self.all_sdf[global_check_idx],
            X_check=X_global_check if X_global_check.size > 0 else None,
            tr_tree=global_tr_tree,
            point_indices=global_check_idx,
        )
        timings["time_shortlist"] += time.perf_counter() - shortlist_t0

        # Temporary diagnostics: counts per SDF band for the local check set
        # and for the shortlist. These entries are added to `info` below
        # when `self.shortlist_diagnostics` is enabled. This helps determine
        # whether OML-only additions arise from shortlist composition or from
        # the eta ranking / greedy selection step.
        if self.shortlist_diagnostics and self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            sdf_local_check = self.all_sdf[local_check_idx]
            tau0 = self.shortlist_band_zero_tol
            tau_pos = self.shortlist_band_pos_tol
            tau_neg = self.shortlist_band_neg_tol
            band_masks_check = [
                np.abs(sdf_local_check) <= tau0,
                (sdf_local_check > tau0) & (sdf_local_check <= tau_pos),
                (sdf_local_check < -tau0) & (sdf_local_check >= -tau_neg),
                sdf_local_check < -tau_neg,
            ]
            shortlist_band_counts_check = [int(np.sum(m)) for m in band_masks_check]

            if local_shortlist.size > 0:
                sdf_local_short = self.all_sdf[local_check_idx[local_shortlist]]
                shortlist_band_counts_short = [
                    int(np.sum(np.abs(sdf_local_short) <= tau0)),
                    int(np.sum((sdf_local_short > tau0) & (sdf_local_short <= tau_pos))),
                    int(np.sum((sdf_local_short < -tau0) & (sdf_local_short >= -tau_neg))),
                    int(np.sum(sdf_local_short < -tau_neg)),
                ]
            else:
                shortlist_band_counts_short = [0, 0, 0, 0]
        else:
            shortlist_band_counts_check = [0, 0, 0, 0]
            shortlist_band_counts_short = [0, 0, 0, 0]

        A_l_lc = None
        shortlist_pred_t0 = time.perf_counter()
        if local_shortlist.size > 0:
            lc_short_global_idx = local_check_idx[local_shortlist]
            background_pred_lc_short = self._evaluate_background_model_on_indices(
                lc_short_global_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )
            X_lc_short = self.all_points[lc_short_global_idx]
            if X_local.shape[0] > 0:
                A_l_lc = assemble_wendland_matrix(
                    X_local,
                    X_lc_short,
                    rho_global,
                    src_tree=local_tr_tree,
                    chunk_size=self.assembly_chunk_size,
                ).tocsr()
                local_pred_lc_short = np.asarray(A_l_lc @ c_local, dtype=float).reshape(-1)
            else:
                local_pred_lc_short = np.zeros((local_shortlist.size,), dtype=float)
            pred_lc_short = background_pred_lc_short + local_pred_lc_short
            r_lc_short = pred_lc_short - self.all_sdf[lc_short_global_idx]
        else:
            r_lc_short = np.zeros((0,), dtype=float)

        if global_shortlist.size > 0:
            gc_short_global_idx = global_check_idx[global_shortlist]
            background_pred_gc_short = self._evaluate_background_model_on_indices(
                gc_short_global_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )
            X_gc_short = self.all_points[gc_short_global_idx]
            if X_local.shape[0] > 0:
                A_l_gc = assemble_wendland_matrix(
                    X_local,
                    X_gc_short,
                    rho_global,
                    src_tree=local_tr_tree,
                    chunk_size=self.assembly_chunk_size,
                )
                local_pred_gc_short = np.asarray(A_l_gc @ c_local, dtype=float).reshape(-1)
            else:
                local_pred_gc_short = np.zeros((global_shortlist.size,), dtype=float)
            pred_gc_short = background_pred_gc_short + local_pred_gc_short
            r_gc_short = pred_gc_short - self.all_sdf[gc_short_global_idx]
        else:
            r_gc_short = np.zeros((0,), dtype=float)
        timings["time_shortlist_prediction"] += time.perf_counter() - shortlist_pred_t0

        local_residual = np.zeros((local_check_idx.size,), dtype=float)
        local_residual[local_shortlist] = r_lc_short

        global_residual = np.zeros((global_check_idx.size,), dtype=float)
        global_residual[global_shortlist] = r_gc_short

        adjoint_t0 = time.perf_counter()
        if self.enable_adjoint and local_factor is not None and A_l_lc is not None and r_lc_short.size > 0:
            w_lc_short = local_weights[local_shortlist]
            adjoint_rhs = np.asarray(A_l_lc.T @ (w_lc_short * r_lc_short), dtype=float).reshape(-1)
            lambda_local = _timed_factor_solve(
                local_factor,
                adjoint_rhs,
                timings=timings,
                key="time_adjoint_solve",
            )
            psi_lc_short = np.asarray(A_l_lc @ lambda_local, dtype=float).reshape(-1)
            eta_lc_short = np.abs(r_lc_short) * np.abs(psi_lc_short)
        else:
            lambda_local = np.zeros((X_local.shape[0],), dtype=float)
            psi_lc_short = np.zeros((local_shortlist.size,), dtype=float)
            eta_lc_short = (
                np.abs(r_lc_short) * local_weights[local_shortlist]
                if local_shortlist.size > 0
                else np.zeros((0,), dtype=float)
            )
        timings["time_adjoint"] += time.perf_counter() - adjoint_t0

        # local_oml_max_abs_error = self._compute_local_oml_max_abs_error(
        #     X_global=X_global,
        #     c_global=c_global,
        #     global_rho=global_rho,
        #     global_tr_tree=global_tr_tree,
        #     X_local=X_local,
        #     c_local=c_local,
        #     rho_global=rho_global,
        #     local_tr_tree=local_tr_tree,
        # )

        local_eta = np.zeros((local_check_idx.size,), dtype=float)
        local_eta[local_shortlist] = self.local_error_weight * eta_lc_short
        local_selection_score = np.zeros((local_check_idx.size,), dtype=float)
        local_selection_terms_short = {
            "feature_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "curvature_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "sign_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "surface_proximity": np.zeros((local_shortlist.size,), dtype=float),
        }
        if local_shortlist.size > 0:
            selection_short, local_selection_terms_short = self._compute_local_selection_scores(
                point_indices=lc_short_global_idx,
                sdf_values=self.all_sdf[lc_short_global_idx],
                pred_values=pred_lc_short,
                eta_base=self.local_error_weight * eta_lc_short,
            )
            local_selection_score[local_shortlist] = selection_short
        global_eta = np.zeros_like(global_residual)

        info = {
            "X_far_field": X_far_field,
            "y_far_field": y_far_field,
            "far_field_center": far_field_center,
            "far_field_reference_radius": far_field_reference_radius,
            "far_field_box_min": far_field_box_min,
            "far_field_box_max": far_field_box_max,
            "local_train_idx": local_train_idx,
            "global_train_idx": global_train_idx,
            "local_check_idx": local_check_idx,
            "global_check_idx": global_check_idx,
            "X_local": X_local,
            "X_global": X_global,
            "c_far_field": c_far_field,
            "y_local": y_local,
            "y_global": y_global,
            "c_local": c_local,
            "c_global": c_global,
            "K_far_field": K_far_field,
            "K_local": K_local,
            "K_global": K_global,
            "far_field_factor": far_field_factor,
            "local_factor": local_factor,
            "global_factor": global_factor,
            "far_field_shape_parameter": far_field_shape_parameter,
            "far_field_length_scale": far_field_length_scale,
            "far_field_blend_inner_radius": far_field_blend_inner_radius,
            "far_field_blend_outer_radius": far_field_blend_outer_radius,
            "far_field_rho": far_field_rho,
            "far_field_blend_inner_distance": far_field_blend_inner_distance,
            "far_field_blend_outer_distance": far_field_blend_outer_distance,
            "rho_local": rho_local,
            "rho_global": rho_global,
            "global_rho": global_rho,
            "local_tr_tree": local_tr_tree,
            "global_tr_tree": global_tr_tree,
            "local_residual": local_residual,
            "global_residual": global_residual,
            "local_weights": local_weights,
            "global_weights": global_weights,
            "local_eta": local_eta,
            "global_eta": global_eta,
            "local_selection_score": local_selection_score,
            "local_shortlist": local_shortlist,
            "global_shortlist": global_shortlist,
            "shortlist_band_counts_check": shortlist_band_counts_check,
            "shortlist_band_counts_shortlist": shortlist_band_counts_short,
            "lambda_local": lambda_local,
            "psi_local_short": psi_lc_short,
            "local_selection_terms_short": local_selection_terms_short,
            "local_oml_max_abs_error": 0.0,
            "timings": timings,
        }

        # Additional diagnostics: L2 errors on OML points and on all training
        # points. Only compute when diagnostics are enabled to avoid overhead.
        if self.shortlist_diagnostics:
            # OML L2 error
            oml_idx = np.asarray(self.oml_point_indices, dtype=int)
            if oml_idx.size > 0:
                oml_eval_t0 = time.perf_counter()
                pred_oml = np.zeros((oml_idx.size,), dtype=float)
                if X_far_field.shape[0] > 0:
                    pred_oml += self._evaluate_far_field_values(
                        self.all_points[oml_idx],
                        X_far_field,
                        c_far_field,
                        far_field_shape_parameter,
                        far_field_blend_inner_radius,
                        far_field_blend_outer_radius,
                    )
                if X_global.shape[0] > 0:
                    pred_oml += self._evaluate_global_model_on_indices(
                        oml_idx,
                        X_global,
                        c_global,
                        global_rho,
                        global_tr_tree,
                    )
                if X_local.shape[0] > 0:
                    A_oml = assemble_wendland_matrix(
                        X_local,
                        self.all_points[oml_idx],
                        rho_global,
                        src_tree=local_tr_tree,
                        chunk_size=self.assembly_chunk_size,
                    ).tocsr()
                    pred_oml += np.asarray(A_oml @ c_local, dtype=float).reshape(-1)
                oml_eval_time = time.perf_counter() - oml_eval_t0
                oml_abs_error = np.abs(pred_oml - self.all_sdf[oml_idx])
                oml_l2 = float(np.sqrt(np.sum(oml_abs_error ** 2)))
                info["local_oml_max_abs_error"] = float(np.max(oml_abs_error)) if oml_abs_error.size > 0 else 0.0
                timings["time_oml_surrogate_eval_total"] = oml_eval_time
                timings["time_oml_surrogate_eval_per_point"] = oml_eval_time / float(oml_idx.size)
            else:
                oml_l2 = 0.0

            # Training L2 error (all current training points)
            train_idx = np.unique(np.concatenate([local_train_idx, global_train_idx])).astype(int)
            if train_idx.size > 0:
                pred_train = np.zeros((train_idx.size,), dtype=float)
                if X_far_field.shape[0] > 0:
                    pred_train += self._evaluate_far_field_values(
                        self.all_points[train_idx],
                        X_far_field,
                        c_far_field,
                        far_field_shape_parameter,
                        far_field_blend_inner_radius,
                        far_field_blend_outer_radius,
                    )
                if X_global.shape[0] > 0:
                    pred_train += self._evaluate_global_model_on_indices(
                        train_idx,
                        X_global,
                        c_global,
                        global_rho,
                        global_tr_tree,
                    )
                if X_local.shape[0] > 0:
                    A_train = assemble_wendland_matrix(
                        X_local,
                        self.all_points[train_idx],
                        rho_global,
                        src_tree=local_tr_tree,
                        chunk_size=self.assembly_chunk_size,
                    ).tocsr()
                    pred_train += np.asarray(A_train @ c_local, dtype=float).reshape(-1)
                train_l2 = float(np.sqrt(np.sum((pred_train - self.all_sdf[train_idx]) ** 2)))
            else:
                train_l2 = 0.0

            info["oml_l2_error"] = oml_l2
            info["train_l2_error"] = train_l2

        timings["time_total_fit"] = time.perf_counter() - total_t0
        self._last_fit_info = info
        return info

    def get_current_fit_info(self, force_refit: bool = False) -> Dict[str, Any]:
        if force_refit or self._last_fit_info is None:
            return self._fit_current_model()
        return self._last_fit_info

    def predict(
        self,
        points: np.ndarray,
        force_refit: bool = False,
    ) -> np.ndarray:
        """
        Evaluate the fitted surrogate at arbitrary query points.

        Uses ``evaluate_rbf_matvec`` which skips sparse-matrix materialization
        and calls ``cKDTree.query_ball_point`` with ``workers=-1`` on the full
        query array in one shot — avoiding the chunked double-loop that made
        this ~20× slower than necessary.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (M, 3)")

        info = self.get_current_fit_info(force_refit=force_refit)
        out = np.zeros((pts.shape[0],), dtype=float)

        if info["X_far_field"].shape[0] > 0:
            if (
                self.freeze_global
                and points is self.all_points
                and self._cached_far_field_values is not None
                and self._cached_far_field_values_mask is not None
            ):
                if np.all(self._cached_far_field_values_mask):
                    out += self._cached_far_field_values.copy()
                else:
                    out += self._evaluate_far_field_on_indices(
                        np.arange(self.n_total, dtype=int),
                        info["X_far_field"],
                        info["c_far_field"],
                        info["far_field_shape_parameter"],
                        info["far_field_blend_inner_radius"],
                        info["far_field_blend_outer_radius"],
                    )
            else:
                out += self._evaluate_far_field_values(
                    pts,
                    info["X_far_field"],
                    info["c_far_field"],
                    info["far_field_shape_parameter"],
                    info["far_field_blend_inner_radius"],
                    info["far_field_blend_outer_radius"],
                )

        if info["X_global"].shape[0] > 0:
            if self.freeze_global and points is self.all_points and self._cached_global_values is not None:
                if self._cached_global_values_mask is not None and np.all(self._cached_global_values_mask):
                    out += self._cached_global_values.copy()
                else:
                    out += self._evaluate_global_model_on_indices(
                        np.arange(self.n_total, dtype=int),
                        info["X_global"],
                        info["c_global"],
                        info["global_rho"],
                        info["global_tr_tree"],
                    )
            else:
                global_op = self._get_or_build_prediction_operator(
                    branch="global",
                    src_pts=info["X_global"],
                    tgt_pts=pts,
                    rho=info["global_rho"],
                    src_tree=info["global_tr_tree"],
                    model_version=self._global_model_version,
                )
                out += np.asarray(global_op @ info["c_global"], dtype=float).reshape(-1)

        if info["X_local"].shape[0] > 0:
            local_op = self._get_or_build_prediction_operator(
                branch="local",
                src_pts=info["X_local"],
                tgt_pts=pts,
                rho=info["rho_global"],
                src_tree=info["local_tr_tree"],
                model_version=self._local_model_version,
            )
            out += np.asarray(local_op @ info["c_local"], dtype=float).reshape(-1)

        return out

    def compute_dwr_scores(self) -> DWRRefinementResult:
        info = self._fit_current_model()

        residual = np.zeros((self.n_total,), dtype=float)
        weights = np.zeros((self.n_total,), dtype=float)
        eta = np.zeros((self.n_total,), dtype=float)
        psi = np.zeros((self.n_total,), dtype=float)

        residual[info["local_check_idx"]] = info["local_residual"]
        residual[info["global_check_idx"]] = info["global_residual"]
        weights[info["local_check_idx"]] = info["local_weights"]
        weights[info["global_check_idx"]] = info["global_weights"]
        eta[info["local_check_idx"]] = info["local_eta"]
        eta[info["global_check_idx"]] = info["global_eta"]
        if info["local_shortlist"].size > 0:
            psi[info["local_check_idx"][info["local_shortlist"]]] = info["psi_local_short"]

        _local_sdf = self.all_sdf[info["local_train_idx"]]
        diagnostics = {
            "n_far_field_train": int(info["X_far_field"].shape[0]),
            "n_local_train": int(info["local_train_idx"].size),
            "n_global_train": int(info["global_train_idx"].size),
            "n_local_train_oml": int(np.sum(np.isclose(_local_sdf, 0.0))),
            "n_local_train_positive": int(np.sum(_local_sdf > 0.0)),
            "n_local_train_negative": int(np.sum(_local_sdf < 0.0)),
            "n_local_check": int(info["local_check_idx"].size),
            "n_global_check": int(info["global_check_idx"].size),
            "far_field_shape_parameter": float(info["far_field_shape_parameter"]),
            "far_field_length_scale": float(info["far_field_length_scale"]),
            "far_field_reference_radius": float(info["far_field_reference_radius"]),
            "far_field_blend_inner_radius": float(info["far_field_blend_inner_radius"]),
            "far_field_blend_outer_radius": float(info["far_field_blend_outer_radius"]),
            "far_field_rho": float(info.get("far_field_rho", info["far_field_shape_parameter"])),
            "far_field_blend_inner_distance": float(
                info.get("far_field_blend_inner_distance", info["far_field_blend_inner_radius"])
            ),
            "far_field_blend_outer_distance": float(
                info.get("far_field_blend_outer_distance", info["far_field_blend_outer_radius"])
            ),
            "rho_global": float(info["rho_global"]),
            "global_rho": float(info["global_rho"]),
            "residual_l2_weighted_local": float(np.sqrt(np.sum(info["local_weights"] * info["local_residual"] ** 2))) if info["local_residual"].size > 0 else 0.0,
            "residual_l2_weighted_global": float(np.sqrt(np.sum(info["global_weights"] * info["global_residual"] ** 2))) if info["global_residual"].size > 0 else 0.0,
            "eta_max_local": float(np.max(info["local_eta"])) if info["local_eta"].size > 0 else 0.0,
            "eta_max_global": float(np.max(info["global_eta"])) if info["global_eta"].size > 0 else 0.0,
            "selection_score_max_local": float(np.max(info["local_selection_score"])) if info["local_selection_score"].size > 0 else 0.0,
            "local_oml_max_abs_error": float(info["local_oml_max_abs_error"]),
            "adjoint_enabled": bool(self.enable_adjoint),
            "selection_score_mode": self.selection_score_mode,
            "geometry_metadata_enabled": bool(self.has_geometry_metadata),
            "hybrid_mode": (
                "sequential_far_field_plus_background_plus_local_optimized"
                if info["X_far_field"].shape[0] > 0
                else "sequential_background_plus_local_optimized"
            ),
        }
        diagnostics.update(info.get("timings", {}))

        return DWRRefinementResult(
            iteration=-1,
            train_indices=np.unique(np.concatenate([info["local_train_idx"], info["global_train_idx"]])).astype(int),
            check_indices=np.unique(np.concatenate([info["local_check_idx"], info["global_check_idx"]])).astype(int),
            rho_local=info["rho_local"],
            rho_global=info["rho_global"],
            residual=residual,
            weights=weights,
            lambda_vec=info["lambda_local"],
            psi=psi,
            eta=eta,
            selected_check_indices=np.array([], dtype=int),
            selected_global_indices=np.array([], dtype=int),
            diagnostics=diagnostics,
        )

    def refine_once(
        self,
        n_add_local: int,
        n_add_global: int = 0,
        min_separation_local: Optional[float] = None,
        min_separation_global: Optional[float] = None,
        iteration: int = -1,
        recompute_scores_after_update: bool = False,
    ) -> DWRRefinementResult:
        info = self._fit_current_model()

        if min_separation_local is None:
            min_separation_local = 0.5 * info["rho_global"]
        if min_separation_global is None:
            min_separation_global = 0.5 * info["global_rho"]

        local_shortlist = info["local_shortlist"]
        selected_local_local = greedy_farthest_subset_from_scores(
            self.all_points[info["local_check_idx"]][local_shortlist],
            info["local_selection_score"][local_shortlist],
            n_select=int(n_add_local),
            min_separation=float(min_separation_local),
        ) if local_shortlist.size > 0 and int(n_add_local) > 0 else np.array([], dtype=int)
        selected_local_global = info["local_check_idx"][local_shortlist[selected_local_local]] if selected_local_local.size > 0 else np.array([], dtype=int)

        if selected_local_global.size > 0:
            self.local_train_mask[selected_local_global] = True
            self._invalidate_local_cache()

        # Temporary diagnostics: counts per SDF band for the selected local points
        if self.shortlist_diagnostics and self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            selected_band_counts = [0, 0, 0, 0]
            if selected_local_global.size > 0:
                sdf_sel = self.all_sdf[selected_local_global]
                tau0 = self.shortlist_band_zero_tol
                tau_pos = self.shortlist_band_pos_tol
                tau_neg = self.shortlist_band_neg_tol
                selected_band_counts = [
                    int(np.sum(np.abs(sdf_sel) <= tau0)),
                    int(np.sum((sdf_sel > tau0) & (sdf_sel <= tau_pos))),
                    int(np.sum((sdf_sel < -tau0) & (sdf_sel >= -tau_neg))),
                    int(np.sum(sdf_sel < -tau_neg)),
                ]
        else:
            selected_band_counts = [0, 0, 0, 0]

        if not self.freeze_global and int(n_add_global) > 0:
            global_shortlist = info["global_shortlist"]
            selected_global_local = greedy_farthest_subset_from_scores(
                self.all_points[info["global_check_idx"]][global_shortlist],
                info["global_eta"][global_shortlist],
                n_select=int(n_add_global),
                min_separation=float(min_separation_global),
            ) if global_shortlist.size > 0 else np.array([], dtype=int)
            selected_global_global = info["global_check_idx"][global_shortlist[selected_global_local]] if selected_global_local.size > 0 else np.array([], dtype=int)
            if selected_global_global.size > 0:
                self.global_train_mask[selected_global_global] = True
                self._invalidate_global_cache()
        else:
            selected_global_global = np.array([], dtype=int)

        if recompute_scores_after_update:
            res = self.compute_dwr_scores()
            res.diagnostics.update(
                {
                    "iteration": int(iteration),
                    "n_added_local": int(selected_local_global.size),
                    "n_added_global": int(selected_global_global.size),
                    "min_separation_local": float(min_separation_local),
                    "min_separation_global": float(min_separation_global),
                    "refit_after_update": True,
                    "shortlist_band_counts_check": info.get("shortlist_band_counts_check", [0,0,0,0]),
                    "shortlist_band_counts_shortlist": info.get("shortlist_band_counts_shortlist", [0,0,0,0]),
                    "selected_band_counts": selected_band_counts,
                    "oml_l2_error": info.get("oml_l2_error", 0.0),
                    "train_l2_error": info.get("train_l2_error", 0.0),
                }
            )
            res.selected_check_indices = np.concatenate([selected_local_global, selected_global_global]).astype(int)
            res.selected_global_indices = res.selected_check_indices.copy()
            res.iteration = iteration
            return res

        residual = np.zeros((self.n_total,), dtype=float)
        weights = np.zeros((self.n_total,), dtype=float)
        eta = np.zeros((self.n_total,), dtype=float)
        psi = np.zeros((self.n_total,), dtype=float)

        residual[info["local_check_idx"]] = info["local_residual"]
        residual[info["global_check_idx"]] = info["global_residual"]
        weights[info["local_check_idx"]] = info["local_weights"]
        weights[info["global_check_idx"]] = info["global_weights"]
        eta[info["local_check_idx"]] = info["local_eta"]
        eta[info["global_check_idx"]] = info["global_eta"]
        if info["local_shortlist"].size > 0:
            psi[info["local_check_idx"][info["local_shortlist"]]] = info["psi_local_short"]

        _local_sdf = self.all_sdf[info["local_train_idx"]]
        diagnostics = {
            "iteration": int(iteration),
            "n_far_field_train": int(info["X_far_field"].shape[0]),
            "n_local_train": int(info["local_train_idx"].size),
            "n_global_train": int(info["global_train_idx"].size),
            "n_local_train_oml": int(np.sum(np.isclose(_local_sdf, 0.0))),
            "n_local_train_positive": int(np.sum(_local_sdf > 0.0)),
            "n_local_train_negative": int(np.sum(_local_sdf < 0.0)),
            "n_local_check": int(info["local_check_idx"].size),
            "n_global_check": int(info["global_check_idx"].size),
            "far_field_shape_parameter": float(info["far_field_shape_parameter"]),
            "far_field_length_scale": float(info["far_field_length_scale"]),
            "far_field_reference_radius": float(info["far_field_reference_radius"]),
            "far_field_blend_inner_radius": float(info["far_field_blend_inner_radius"]),
            "far_field_blend_outer_radius": float(info["far_field_blend_outer_radius"]),
            "far_field_rho": float(info.get("far_field_rho", info["far_field_shape_parameter"])),
            "far_field_blend_inner_distance": float(
                info.get("far_field_blend_inner_distance", info["far_field_blend_inner_radius"])
            ),
            "far_field_blend_outer_distance": float(
                info.get("far_field_blend_outer_distance", info["far_field_blend_outer_radius"])
            ),
            "rho_global": float(info["rho_global"]),
            "global_rho": float(info["global_rho"]),
            "residual_l2_weighted_local": float(np.sqrt(np.sum(info["local_weights"] * info["local_residual"] ** 2))) if info["local_residual"].size > 0 else 0.0,
            "residual_l2_weighted_global": float(np.sqrt(np.sum(info["global_weights"] * info["global_residual"] ** 2))) if info["global_residual"].size > 0 else 0.0,
            "eta_max_local": float(np.max(info["local_eta"])) if info["local_eta"].size > 0 else 0.0,
            "eta_max_global": float(np.max(info["global_eta"])) if info["global_eta"].size > 0 else 0.0,
            "selection_score_max_local": float(np.max(info["local_selection_score"])) if info["local_selection_score"].size > 0 else 0.0,
            "local_oml_max_abs_error": float(info["local_oml_max_abs_error"]),
            "n_added_local": int(selected_local_global.size),
            "n_added_global": int(selected_global_global.size),
            "min_separation_local": float(min_separation_local),
            "min_separation_global": float(min_separation_global),
            "refit_after_update": False,
            "shortlist_band_counts_check": info.get("shortlist_band_counts_check", [0,0,0,0]),
            "shortlist_band_counts_shortlist": info.get("shortlist_band_counts_shortlist", [0,0,0,0]),
            "selected_band_counts": selected_band_counts,
            "oml_l2_error": info.get("oml_l2_error", 0.0),
            "train_l2_error": info.get("train_l2_error", 0.0),
            "adjoint_enabled": bool(self.enable_adjoint),
            "selection_score_mode": self.selection_score_mode,
            "geometry_metadata_enabled": bool(self.has_geometry_metadata),
            "hybrid_mode": (
                "sequential_far_field_plus_background_plus_local_optimized"
                if info["X_far_field"].shape[0] > 0
                else "sequential_background_plus_local_optimized"
            ),
        }
        diagnostics.update(info.get("timings", {}))

        return DWRRefinementResult(
            iteration=iteration,
            train_indices=np.unique(np.concatenate([info["local_train_idx"], info["global_train_idx"]])).astype(int),
            check_indices=np.unique(np.concatenate([info["local_check_idx"], info["global_check_idx"]])).astype(int),
            rho_local=info["rho_local"],
            rho_global=info["rho_global"],
            residual=residual,
            weights=weights,
            lambda_vec=info["lambda_local"],
            psi=psi,
            eta=eta,
            selected_check_indices=np.concatenate([selected_local_global, selected_global_global]).astype(int),
            selected_global_indices=np.concatenate([selected_local_global, selected_global_global]).astype(int),
            diagnostics=diagnostics,
        )

    def refine(
        self,
        n_iterations: int,
        n_add_local_per_iter: int,
        n_add_global_per_iter: int = 0,
        min_separation_local: Optional[float] = None,
        min_separation_global: Optional[float] = None,
        verbose: bool = True,
        recompute_scores_after_update: bool = False,
    ) -> list[DWRRefinementResult]:
        history = []
        for it in range(n_iterations):
            res = self.refine_once(
                n_add_local=n_add_local_per_iter,
                n_add_global=n_add_global_per_iter,
                min_separation_local=min_separation_local,
                min_separation_global=min_separation_global,
                iteration=it,
                recompute_scores_after_update=recompute_scores_after_update,
            )
            history.append(res)
            if verbose:
                d = res.diagnostics
                extra = ""
                if self.shortlist_diagnostics:
                    oml = d.get("oml_l2_error")
                    train = d.get("train_l2_error")
                    if oml is not None and train is not None:
                        extra = f" oml_L2={oml:.3e} train_L2={train:.3e}"

                if self.freeze_global:
                    s = (
                        f"[hybrid-opt iter {it:02d}] "
                        f"n_local_train={d['n_local_train']:6d} "
                        f"(oml={d['n_local_train_oml']:5d} "
                        f"pos={d['n_local_train_positive']:5d} "
                        f"neg={d['n_local_train_negative']:5d}) "
                        f"added_local={d['n_added_local']:4d} "
                        f"selected_split={d.get('selected_band_counts', [0,0,0,0])} "
                        f"{extra} "
                        f"||r||_W,local={d['residual_l2_weighted_local']:.4e}"
                    )
                    print(s)
                else:
                    s = (
                        f"[hybrid-opt iter {it:02d}] "
                        f"n_local_train={d['n_local_train']:6d} "
                        f"(oml={d['n_local_train_oml']:5d} "
                        f"pos={d['n_local_train_positive']:5d} "
                        f"neg={d['n_local_train_negative']:5d}) "
                        f"added_local={d['n_added_local']:4d} "
                        f"added_global={d['n_added_global']:4d} "
                        f"selected_split={d.get('selected_band_counts', [0,0,0,0])} "
                        f"{extra} "
                        f"||r||_W,local={d['residual_l2_weighted_local']:.4e} "
                        f"||r||_W,global={d['residual_l2_weighted_global']:.4e}"
                    )
                    print(s)
        return history


class HybridWendlandC2DWRRefinerIncremental(HybridWendlandC2DWRRefinerOptimized):
    """
    Incremental drop-in replacement for ``HybridWendlandC2DWRRefinerOptimized``.

    The local branch keeps an append-only center ordering and reuses a staged
    support radius so that small batches of newly selected local points can be
    absorbed via Schur-complement block updates instead of rebuilding the full
    local factorization every iteration.
    """

    def __init__(
        self,
        all_points: np.ndarray,
        all_sdf: np.ndarray,
        initial_local_train_indices: np.ndarray,
        initial_global_train_indices: np.ndarray,
        far_field_points: Optional[np.ndarray] = None,
        far_field_sdf: Optional[np.ndarray] = None,
        far_field_box_min: Optional[np.ndarray] = None,
        far_field_box_max: Optional[np.ndarray] = None,
        oml_point_indices: Optional[np.ndarray] = None,
        local_candidate_mask: Optional[np.ndarray] = None,
        global_candidate_mask: Optional[np.ndarray] = None,
        point_metadata: Optional[Dict[str, np.ndarray]] = None,
        neighbor_k: int = 24,
        radius_scale: float = 1.5,
        radius_quantile: float = 0.90,
        radius_inflate: float = 1.10,
        weight_power: float = 1.0,
        weight_eps: float = 1e-3,
        jitter: float = 1e-10,
        shortlist_size: Optional[int] = None,
        shortlist_fraction: float = 0.1,
        assembly_chunk_size: int = 4096,
        global_max_centers: Optional[int] = 5000,
        global_radius_scale: float = 5.0,
        global_radius_quantile: float = 0.90,
        global_radius_inflate: float = 1.10,
        global_ridge: float = 1e-8,
        far_field_neighbor_k: int = 12,
        far_field_radius_scale: float = 1.75,
        far_field_radius_quantile: float = 0.90,
        far_field_radius_inflate: float = 1.10,
        far_field_ridge: float = 1e-10,
        far_field_eval_chunk_size: int = 2048,
        far_field_blend_inner_distance: Optional[float] = None,
        far_field_blend_outer_distance: Optional[float] = None,
        local_error_weight: float = 1.0,
        freeze_global: bool = True,
        enable_adjoint: bool = True,
        shortlist_mode: str = "weight",
        shortlist_dist_gamma: float = 1.0,
        shortlist_band_zero_tol: float = 5e-3,
        shortlist_band_pos_tol: float = 5e-2,
        shortlist_band_neg_tol: float = 5e-2,
        shortlist_band_weights: Optional[Tuple[float, float, float, float]] = None,
        shortlist_score_clip: float = 50.0,
        shortlist_dist_beta: float = 0.35,
        shortlist_feature_alpha: float = 1.0,
        shortlist_feature_fraction: float = 0.25,
        shortlist_feature_tau: float = 5e-2,
        selection_score_mode: str = "eta",
        selection_feature_weight: float = 0.35,
        selection_curvature_weight: float = 0.25,
        selection_sign_weight: float = 0.5,
        selection_surface_tau: float = 5e-2,
        shortlist_diagnostics: bool = False,
    ):
        super().__init__(
            all_points=all_points,
            all_sdf=all_sdf,
            initial_local_train_indices=initial_local_train_indices,
            initial_global_train_indices=initial_global_train_indices,
            far_field_points=far_field_points,
            far_field_sdf=far_field_sdf,
            far_field_box_min=far_field_box_min,
            far_field_box_max=far_field_box_max,
            oml_point_indices=oml_point_indices,
            local_candidate_mask=local_candidate_mask,
            global_candidate_mask=global_candidate_mask,
            point_metadata=point_metadata,
            neighbor_k=neighbor_k,
            radius_scale=radius_scale,
            radius_quantile=radius_quantile,
            radius_inflate=radius_inflate,
            weight_power=weight_power,
            weight_eps=weight_eps,
            jitter=jitter,
            shortlist_size=shortlist_size,
            shortlist_fraction=shortlist_fraction,
            assembly_chunk_size=assembly_chunk_size,
            global_max_centers=global_max_centers,
            global_radius_scale=global_radius_scale,
            global_radius_quantile=global_radius_quantile,
            global_radius_inflate=global_radius_inflate,
            global_ridge=global_ridge,
            far_field_neighbor_k=far_field_neighbor_k,
            far_field_radius_scale=far_field_radius_scale,
            far_field_radius_quantile=far_field_radius_quantile,
            far_field_radius_inflate=far_field_radius_inflate,
            far_field_ridge=far_field_ridge,
            far_field_eval_chunk_size=far_field_eval_chunk_size,
            far_field_blend_inner_distance=far_field_blend_inner_distance,
            far_field_blend_outer_distance=far_field_blend_outer_distance,
            local_error_weight=local_error_weight,
            freeze_global=freeze_global,
            enable_adjoint=enable_adjoint,
            shortlist_mode=shortlist_mode,
            shortlist_dist_gamma=shortlist_dist_gamma,
            shortlist_band_zero_tol=shortlist_band_zero_tol,
            shortlist_band_pos_tol=shortlist_band_pos_tol,
            shortlist_band_neg_tol=shortlist_band_neg_tol,
            shortlist_band_weights=shortlist_band_weights,
            shortlist_score_clip=shortlist_score_clip,
            shortlist_dist_beta=shortlist_dist_beta,
            shortlist_feature_alpha=shortlist_feature_alpha,
            shortlist_feature_fraction=shortlist_feature_fraction,
            shortlist_feature_tau=shortlist_feature_tau,
            selection_score_mode=selection_score_mode,
            selection_feature_weight=selection_feature_weight,
            selection_curvature_weight=selection_curvature_weight,
            selection_sign_weight=selection_sign_weight,
            selection_surface_tau=selection_surface_tau,
            shortlist_diagnostics=shortlist_diagnostics,
        )

        self._local_train_indices_ordered = ordered_unique_indices(initial_local_train_indices)
        self.local_train_mask[:] = False
        self.local_train_mask[self._local_train_indices_ordered] = True

        self._local_state: Optional[_IncrementalLocalState] = None
        self._distance_to_train_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

        self._local_radius_policy = "stage"
        self._local_rebuild_every = 5
        self._local_radius_growth_tol = 0.10
        self._incremental_min_old_size = 5000
        self._incremental_max_batch_size = 5000
        self._incremental_max_relative_batch = 0.20
        self._incremental_rhs_block_size = 64
        self._schur_jitter = max(float(self.jitter), 1e-12)

        self._force_local_rebuild_next = False
        self._disable_incremental_local_update = False
        self._debug_force_incremental_failure = False

    def _clear_branch_caches(self, branch: str) -> None:
        self._prediction_operator_cache = {
            k: v for k, v in self._prediction_operator_cache.items() if k[0] != branch
        }
        self._distance_to_train_cache = {
            k: v for k, v in self._distance_to_train_cache.items() if k[0] != branch
        }

    def _invalidate_local_cache(self) -> None:
        self._local_model_version += 1
        self._clear_branch_caches("local")
        self._invalidate_fit_cache()

    def _invalidate_global_cache(self) -> None:
        super()._invalidate_global_cache()
        self._distance_to_train_cache = {
            k: v for k, v in self._distance_to_train_cache.items() if k[0] != "global"
        }

    def _current_indices(self) -> Dict[str, np.ndarray]:
        ordered_local = self._local_train_indices_ordered[
            self.local_train_mask[self._local_train_indices_ordered]
        ]
        remaining_mask = self.local_train_mask.copy()
        remaining_mask[ordered_local] = False
        extra_local = np.where(remaining_mask)[0].astype(np.int64, copy=False)
        if extra_local.size > 0:
            ordered_local = np.concatenate([ordered_local, extra_local]).astype(np.int64, copy=False)
        self._local_train_indices_ordered = ordered_local

        global_train_idx = np.where(self.global_train_mask)[0]
        local_check_mask = self.local_candidate_mask & (~self.local_train_mask)
        if self.freeze_global:
            global_check_idx = np.array([], dtype=np.int64)
        else:
            global_check_mask = self.global_candidate_mask & (~self.global_train_mask)
            global_check_idx = np.where(global_check_mask)[0]

        return {
            "local_train_idx": ordered_local,
            "global_train_idx": global_train_idx,
            "local_check_idx": np.where(local_check_mask)[0],
            "global_check_idx": global_check_idx,
        }

    def _append_local_train_indices(self, new_indices: np.ndarray) -> np.ndarray:
        idx = ordered_unique_indices(new_indices)
        if idx.size == 0:
            return idx
        idx = idx[self.local_candidate_mask[idx]]
        idx = idx[~self.local_train_mask[idx]]
        if idx.size == 0:
            return idx
        self.local_train_mask[idx] = True
        self._local_train_indices_ordered = np.concatenate(
            [self._local_train_indices_ordered, idx]
        ).astype(np.int64, copy=False)
        self._invalidate_local_cache()
        return idx

    def _compute_shortlist_indices(
        self,
        scores: np.ndarray,
        sdf_values: Optional[np.ndarray] = None,
        X_check: Optional[np.ndarray] = None,
        tr_tree: Optional[cKDTree] = None,
        point_indices: Optional[np.ndarray] = None,
        branch: str = "local",
    ) -> np.ndarray:
        n = scores.size
        if n == 0:
            return np.array([], dtype=int)
        shortlist_n = self.shortlist_size
        if shortlist_n is None:
            shortlist_n = int(np.ceil(self.shortlist_fraction * n))
        shortlist_n = max(1, min(int(shortlist_n), n))
        if shortlist_n >= n:
            return np.arange(n, dtype=int)

        d_to_train = None
        if X_check is not None and tr_tree is not None:
            if point_indices is not None:
                model_version = self._local_model_version if branch == "local" else self._global_model_version
                d_to_train = self._get_or_build_distance_to_train_cache(
                    branch=branch,
                    point_indices=point_indices,
                    X_check=X_check,
                    tr_tree=tr_tree,
                    model_version=model_version,
                )
            else:
                d_to_train, _ = tr_tree.query(X_check, k=1, workers=-1)
                d_to_train = np.maximum(np.asarray(d_to_train, dtype=float).reshape(-1), 0.0)

        if self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            if sdf_values is None:
                raise ValueError(
                    "sdf_values are required for shortlist_mode in "
                    "{'banded_weight', 'geometry_banded_weight'}"
                )

            sdf_values = np.asarray(sdf_values, dtype=float).reshape(-1)
            if sdf_values.shape[0] != n:
                raise ValueError("sdf_values must have the same length as scores")

            if d_to_train is not None:
                d_ref = float(np.median(d_to_train)) if d_to_train.size > 0 else 1.0
                if not np.isfinite(d_ref) or d_ref <= 0.0:
                    positive = d_to_train[d_to_train > 0.0]
                    d_ref = float(np.median(positive)) if positive.size > 0 else 1.0
                coverage = 1.0 + self.shortlist_dist_beta * (
                    d_to_train / max(d_ref, 1e-12)
                ) ** self.shortlist_dist_gamma
            else:
                coverage = np.ones((n,), dtype=float)

            clipped_scores = np.minimum(np.asarray(scores, dtype=float), self.shortlist_score_clip)
            combined_scores = clipped_scores * coverage
            geometry_feature_scores = np.zeros((n,), dtype=float)
            use_geometry_shortlist = (
                self.shortlist_mode == "geometry_banded_weight"
                and point_indices is not None
                and self.has_geometry_metadata
            )
            if use_geometry_shortlist:
                geometry_feature_scores = (
                    surface_proximity_weights(sdf_values, self.shortlist_feature_tau)
                    * self._geometry_feature_for_indices(np.asarray(point_indices, dtype=int))
                )
                combined_scores = combined_scores * (
                    1.0 + self.shortlist_feature_alpha * geometry_feature_scores
                )

            tau0 = self.shortlist_band_zero_tol
            tau_pos = self.shortlist_band_pos_tol
            tau_neg = self.shortlist_band_neg_tol
            band_masks = [
                np.abs(sdf_values) <= tau0,
                (sdf_values > tau0) & (sdf_values <= tau_pos),
                (sdf_values < -tau0) & (sdf_values >= -tau_neg),
                sdf_values < -tau_neg,
            ]
            band_indices = [np.flatnonzero(mask) for mask in band_masks]
            quotas_float = np.asarray(self.shortlist_band_weights, dtype=float) * float(shortlist_n)
            quotas = np.floor(quotas_float).astype(int)
            remainder = int(shortlist_n - np.sum(quotas))
            if remainder > 0:
                frac_order = np.argsort(-(quotas_float - quotas))
                for idx in frac_order[:remainder]:
                    quotas[idx] += 1

            selected_parts = []
            selected_total = 0
            leftover = 0
            for band_id in [0, 1, 2, 3]:
                idx_band = band_indices[band_id]
                target = quotas[band_id] + leftover
                if idx_band.size == 0 or target <= 0:
                    leftover = max(target, 0)
                    continue
                take = min(int(target), int(idx_band.size))
                chosen_parts = []

                if use_geometry_shortlist:
                    band_feature_scores = geometry_feature_scores[idx_band]
                    positive_feature = band_feature_scores > 0.0
                    feature_take = 0
                    if np.any(positive_feature) and self.shortlist_feature_fraction > 0.0:
                        feature_take = int(np.round(self.shortlist_feature_fraction * float(take)))
                        if feature_take == 0 and take > 0:
                            feature_take = 1
                        feature_take = min(
                            int(take),
                            int(np.count_nonzero(positive_feature)),
                            int(feature_take),
                        )
                    if feature_take > 0:
                        feat_idx_local = top_k_sorted_indices(band_feature_scores, feature_take)
                        if feat_idx_local.size > 0:
                            chosen_parts.append(idx_band[feat_idx_local])

                already_chosen = (
                    np.unique(np.concatenate(chosen_parts)).astype(int, copy=False)
                    if chosen_parts
                    else np.zeros((0,), dtype=int)
                )
                remaining_band_mask = np.ones((idx_band.size,), dtype=bool)
                if already_chosen.size > 0:
                    remaining_band_mask[np.isin(idx_band, already_chosen)] = False
                remaining_band = idx_band[remaining_band_mask]
                remaining_take = int(take - already_chosen.size)
                if remaining_take > 0 and remaining_band.size > 0:
                    proxy_idx_local = top_k_sorted_indices(combined_scores[remaining_band], remaining_take)
                    if proxy_idx_local.size > 0:
                        chosen_parts.append(remaining_band[proxy_idx_local])

                chosen = (
                    np.unique(np.concatenate(chosen_parts)).astype(int, copy=False)
                    if chosen_parts
                    else np.zeros((0,), dtype=int)
                )
                if chosen.size > 0:
                    order = np.argsort(combined_scores[chosen])[::-1]
                    chosen = chosen[order]
                selected_parts.append(chosen)
                selected_total += chosen.size
                leftover = int(target - chosen.size)

            if selected_total < shortlist_n:
                already = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
                remaining_mask = np.ones((n,), dtype=bool)
                remaining_mask[already] = False
                remaining_idx = np.flatnonzero(remaining_mask)
                if remaining_idx.size > 0:
                    need = min(int(shortlist_n - selected_total), int(remaining_idx.size))
                    rem_short = top_k_sorted_indices(combined_scores[remaining_idx], need)
                    selected_parts.append(remaining_idx[rem_short])

            shortlist = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=int)
            if shortlist.size == 0:
                return shortlist
            shortlist_scores = combined_scores[shortlist]
            return shortlist[np.argsort(shortlist_scores)[::-1]].astype(int, copy=False)

        if self.shortlist_mode == "dist_x_weight" and d_to_train is not None:
            combined_scores = scores * (d_to_train ** self.shortlist_dist_gamma)
        else:
            combined_scores = scores

        shortlist = top_k_sorted_indices(combined_scores, shortlist_n)
        return shortlist[np.argsort(combined_scores[shortlist])[::-1]].astype(int, copy=False)

    def _get_or_build_distance_to_train_cache(
        self,
        branch: str,
        point_indices: np.ndarray,
        X_check: np.ndarray,
        tr_tree: Optional[cKDTree],
        model_version: int,
    ) -> np.ndarray:
        idx = np.asarray(point_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0 or X_check.size == 0 or tr_tree is None:
            return np.zeros((idx.size,), dtype=float)

        key = (branch, hash_index_array(idx), int(model_version))
        cached = self._distance_to_train_cache.get(key)
        if cached is not None:
            return cached.copy()

        d_to_train, _ = tr_tree.query(X_check, k=1, workers=-1)
        d_to_train = np.maximum(np.asarray(d_to_train, dtype=float).reshape(-1), 0.0)
        self._distance_to_train_cache[key] = d_to_train
        return d_to_train.copy()

    def _evaluate_global_model_on_indices(
        self,
        point_indices: np.ndarray,
        X_global: np.ndarray,
        c_global: np.ndarray,
        global_rho: float,
        global_tr_tree: Optional[cKDTree],
    ) -> np.ndarray:
        point_indices = np.asarray(point_indices, dtype=int)
        if point_indices.size == 0 or X_global.shape[0] == 0:
            return np.zeros((point_indices.size,), dtype=float)

        if self.freeze_global and self._cached_global_values is not None:
            assert self._cached_global_values_mask is not None
            if np.all(self._cached_global_values_mask[point_indices]):
                return self._cached_global_values[point_indices].copy()

        pred = evaluate_rbf_matvec(
            X_global,
            self.all_points[point_indices],
            global_rho,
            c_global,
            src_tree=global_tr_tree,
        )
        if self.freeze_global:
            if self._cached_global_values is None:
                self._cached_global_values = np.zeros((self.n_total,), dtype=float)
                self._cached_global_values_mask = np.zeros((self.n_total,), dtype=bool)
            self._cached_global_values[point_indices] = pred
            self._cached_global_values_mask[point_indices] = True
        return pred

    def _evaluate_local_values(
        self,
        points: np.ndarray,
        X_local: np.ndarray,
        c_local: np.ndarray,
        rho_global: float,
        local_tr_tree: Optional[cKDTree],
    ) -> np.ndarray:
        if points.shape[0] == 0 or X_local.shape[0] == 0:
            return np.zeros((points.shape[0],), dtype=float)
        return evaluate_rbf_matvec(
            X_local,
            points,
            rho_global,
            c_local,
            src_tree=local_tr_tree,
        )

    def _evaluate_local_operator(
        self,
        X_local: np.ndarray,
        points: np.ndarray,
        rho_global: float,
        local_tr_tree: Optional[cKDTree],
    ) -> sparse.csr_matrix:
        return self._get_or_build_prediction_operator(
            branch="local",
            src_pts=X_local,
            tgt_pts=points,
            rho=rho_global,
            src_tree=local_tr_tree,
            model_version=self._local_model_version,
        )

    def _maybe_bump_local_structure_version(
        self,
        existing_state: Optional[_IncrementalLocalState],
        new_train_indices: np.ndarray,
        new_rho: float,
    ) -> None:
        if existing_state is None:
            return
        structure_changed = (
            not np.array_equal(existing_state.train_indices, new_train_indices)
            or not np.isclose(existing_state.rho_global, new_rho)
        )
        if structure_changed and existing_state.model_version == self._local_model_version:
            self._local_model_version += 1
            self._clear_branch_caches("local")

    def _should_use_incremental_local_update(
        self,
        state: Optional[_IncrementalLocalState],
        local_train_idx: np.ndarray,
        rho_local: np.ndarray,
    ) -> Tuple[bool, str]:
        if state is None:
            return False, "no_state"
        if self._disable_incremental_local_update:
            return False, "incremental_disabled"
        if self._force_local_rebuild_next:
            return False, "forced_rebuild"
        if state.factor is None:
            return False, "missing_factor"

        old_n = int(state.train_indices.size)
        new_n = int(local_train_idx.size - old_n)
        if new_n <= 0:
            return False, "no_new_points"
        if local_train_idx.size < old_n:
            return False, "train_set_shrank"
        if not np.array_equal(local_train_idx[:old_n], state.train_indices):
            return False, "non_append_update"
        if old_n < int(self._incremental_min_old_size):
            return False, "old_block_too_small"
        if new_n > int(self._incremental_max_batch_size):
            return False, "batch_too_large"
        if old_n > 0 and (new_n / old_n) > float(self._incremental_max_relative_batch):
            return False, "relative_batch_too_large"
        if state.n_updates_since_rebuild >= int(self._local_rebuild_every):
            return False, "scheduled_rebuild"

        suggested_rho = (
            robust_global_support_radius(
                rho_local,
                quantile=self.radius_quantile,
                inflate=self.radius_inflate,
            )
            if rho_local.size > 0
            else 1.0
        )
        if (
            self._local_radius_policy == "stage"
            and suggested_rho > (1.0 + float(self._local_radius_growth_tol)) * float(state.rho_global)
        ):
            return False, "radius_growth"

        new_indices = local_train_idx[old_n:]
        if new_indices.size == 0:
            return False, "empty_batch"
        new_points = self.all_points[new_indices]
        tol = max(1e-12, 1e-10 * max(float(state.rho_global), 1.0))
        if state.tree is not None:
            d_old, _ = state.tree.query(new_points, k=1, workers=-1)
            if np.any(np.asarray(d_old, dtype=float).reshape(-1) <= tol):
                return False, "near_duplicate_to_existing"
        if new_points.shape[0] > 1:
            new_tree = cKDTree(new_points)
            d_new, _ = new_tree.query(new_points, k=min(2, new_points.shape[0]), workers=-1)
            d_new = np.asarray(d_new, dtype=float)
            if d_new.ndim == 2 and d_new.shape[1] > 1 and np.any(d_new[:, 1] <= tol):
                return False, "near_duplicate_in_batch"
        return True, "incremental"

    def _build_local_state_full(
        self,
        local_train_idx: np.ndarray,
        X_local: np.ndarray,
        y_local_residual: np.ndarray,
        rho_local: np.ndarray,
        rebuild_reason: str,
        existing_state: Optional[_IncrementalLocalState],
        timings: Dict[str, float],
    ) -> _IncrementalLocalState:
        fit_t0 = time.perf_counter()
        rho_global = (
            robust_global_support_radius(
                rho_local,
                quantile=self.radius_quantile,
                inflate=self.radius_inflate,
            )
            if rho_local.size > 0
            else 1.0
        )
        self._maybe_bump_local_structure_version(existing_state, local_train_idx, rho_global)

        local_tr_tree = cKDTree(X_local) if X_local.shape[0] > 0 else None
        assembly_t0 = time.perf_counter()
        K_local = assemble_wendland_matrix(
            X_local,
            X_local,
            rho_global,
            src_tree=local_tr_tree,
            chunk_size=self.assembly_chunk_size,
        ).tocsc()
        K_local = K_local + self.jitter * sparse.eye(K_local.shape[0], format="csc")
        timings["time_local_matrix_assembly"] += time.perf_counter() - assembly_t0

        local_factor = _timed_factorize_spd(
            K_local,
            timings=timings,
            key="time_local_factorization",
        )

        coeffs = _timed_factor_solve(
            local_factor,
            y_local_residual,
            timings=timings,
            key="time_local_solve",
        ).reshape(-1)
        timings["time_fit_local_full"] += time.perf_counter() - fit_t0
        return _IncrementalLocalState(
            train_indices=np.asarray(local_train_idx, dtype=np.int64).copy(),
            X=np.asarray(X_local, dtype=float).copy(),
            y_residual=np.asarray(y_local_residual, dtype=float).copy(),
            rho_local=np.asarray(rho_local, dtype=float).copy(),
            rho_global=float(rho_global),
            tree=local_tr_tree,
            K=K_local,
            factor=local_factor,
            coeffs=coeffs,
            n_rebuilds=1 if existing_state is None else int(existing_state.n_rebuilds) + 1,
            n_updates_since_rebuild=0,
            model_version=int(self._local_model_version),
            last_update_mode="full",
            last_rebuild_reason=rebuild_reason,
        )

    def _solve_local_schur_update(
        self,
        state: _IncrementalLocalState,
        X_new: np.ndarray,
        y_old: np.ndarray,
        y_new: np.ndarray,
        timings: Dict[str, float],
    ) -> Tuple[Any, np.ndarray]:
        if self._debug_force_incremental_failure:
            raise RuntimeError("debug incremental failure requested")

        schur_t0 = time.perf_counter()
        new_tree = cKDTree(X_new) if X_new.shape[0] > 0 else None
        K_no = assemble_wendland_matrix(
            state.X,
            X_new,
            state.rho_global,
            src_tree=state.tree,
            chunk_size=self.assembly_chunk_size,
        ).tocsr()
        K_on = K_no.T.tocsc()
        K_nn = assemble_wendland_matrix(
            X_new,
            X_new,
            state.rho_global,
            src_tree=new_tree,
            chunk_size=self.assembly_chunk_size,
        ).tocsc()
        K_nn = K_nn + self.jitter * sparse.eye(K_nn.shape[0], format="csc")

        v_old = _timed_factor_solve(
            state.factor,
            y_old,
            timings=timings,
            key="time_local_solve",
        ).reshape(-1)
        rhs_new = np.asarray(y_new - K_no @ v_old, dtype=float).reshape(-1)
        S = np.asarray(K_nn.toarray(), dtype=float)

        batch_size = max(1, min(int(self._incremental_rhs_block_size), X_new.shape[0]))
        for start in range(0, X_new.shape[0], batch_size):
            stop = min(start + batch_size, X_new.shape[0])
            rhs_block = np.asarray(K_on[:, start:stop].toarray(), dtype=float)
            U_block = _timed_factor_solve(
                state.factor,
                rhs_block,
                timings=timings,
                key="time_local_solve",
            )
            S[:, start:stop] -= np.asarray(K_no @ U_block, dtype=float)

        S = 0.5 * (S + S.T)
        if S.shape[0] > 0:
            S[np.diag_indices_from(S)] += self._schur_jitter
        timings["time_local_schur_assembly"] += time.perf_counter() - schur_t0

        schur_factor = _timed_factorize_spd(
            S,
            timings=timings,
            key="time_local_schur_factorization",
        )

        c_new = _timed_factor_solve(
            schur_factor,
            rhs_new,
            timings=timings,
            key="time_local_solve",
        ).reshape(-1)
        correction_rhs = np.asarray(K_on @ c_new, dtype=float).reshape(-1)
        c_old = v_old - _timed_factor_solve(
            state.factor,
            correction_rhs,
            timings=timings,
            key="time_local_solve",
        ).reshape(-1)
        coeffs = np.concatenate([c_old, c_new]).astype(float, copy=False)
        return _BlockSchurFactor(state.factor, K_on, K_no, schur_factor), coeffs

    def _update_local_state_incremental(
        self,
        state: _IncrementalLocalState,
        local_train_idx: np.ndarray,
        X_local: np.ndarray,
        y_local_residual: np.ndarray,
        rho_local: np.ndarray,
        timings: Dict[str, float],
    ) -> _IncrementalLocalState:
        fit_t0 = time.perf_counter()
        old_n = int(state.train_indices.size)
        X_new = X_local[old_n:]
        y_old = y_local_residual[:old_n]
        y_new = y_local_residual[old_n:]
        factor, coeffs = self._solve_local_schur_update(
            state=state,
            X_new=X_new,
            y_old=y_old,
            y_new=y_new,
            timings=timings,
        )
        timings["time_fit_local_incremental"] += time.perf_counter() - fit_t0
        return _IncrementalLocalState(
            train_indices=np.asarray(local_train_idx, dtype=np.int64).copy(),
            X=np.asarray(X_local, dtype=float).copy(),
            y_residual=np.asarray(y_local_residual, dtype=float).copy(),
            rho_local=np.asarray(rho_local, dtype=float).copy(),
            rho_global=float(state.rho_global),
            tree=cKDTree(X_local),
            K=None,
            factor=factor,
            coeffs=np.asarray(coeffs, dtype=float).reshape(-1),
            n_rebuilds=int(state.n_rebuilds),
            n_updates_since_rebuild=int(state.n_updates_since_rebuild) + 1,
            model_version=int(self._local_model_version),
            last_update_mode="incremental",
            last_rebuild_reason=state.last_rebuild_reason,
        )

    def _refresh_local_residual_targets(
        self,
        state: _IncrementalLocalState,
        y_local_residual: np.ndarray,
        timings: Optional[Dict[str, float]] = None,
    ) -> _IncrementalLocalState:
        coeffs = _timed_factor_solve(
            state.factor,
            y_local_residual,
            timings=timings,
            key="time_local_solve",
        ).reshape(-1)
        state.y_residual = np.asarray(y_local_residual, dtype=float).copy()
        state.coeffs = coeffs
        state.rho_local = state.rho_local.copy()
        state.last_update_mode = "reuse"
        return state

    def _fit_current_model(self) -> Dict[str, Any]:
        total_t0 = time.perf_counter()
        timings = {
            "time_fit_global": 0.0,
            "time_fit_local_full": 0.0,
            "time_fit_local_incremental": 0.0,
            "time_local_matrix_assembly": 0.0,
            "time_local_factorization": 0.0,
            "time_local_schur_assembly": 0.0,
            "time_local_schur_factorization": 0.0,
            "time_shortlist": 0.0,
            "time_shortlist_prediction": 0.0,
            "time_adjoint": 0.0,
            "time_total_fit": 0.0,
            "time_factorization_total": 0.0,
            "time_solve_total": 0.0,
            "time_global_factorization": 0.0,
            "time_global_solve": 0.0,
            "time_local_solve": 0.0,
            "time_adjoint_solve": 0.0,
            "time_oml_surrogate_eval_total": 0.0,
            "time_oml_surrogate_eval_per_point": 0.0,
        }

        idx = self._current_indices()
        local_train_idx = idx["local_train_idx"]
        global_train_idx = idx["global_train_idx"]
        local_check_idx = idx["local_check_idx"]
        global_check_idx = idx["global_check_idx"]

        X_local = self.all_points[local_train_idx]
        y_local = self.all_sdf[local_train_idx]
        X_global = self.all_points[global_train_idx]
        y_global = self.all_sdf[global_train_idx]
        far_field_info = self._fit_far_field_stage()
        X_far_field = far_field_info["X_far_field"]
        y_far_field = far_field_info["y_far_field"]
        K_far_field = far_field_info["K_far_field"]
        far_field_factor = far_field_info["far_field_factor"]
        c_far_field = far_field_info["c_far_field"]
        far_field_center = far_field_info["far_field_center"]
        far_field_reference_radius = far_field_info["far_field_reference_radius"]
        far_field_shape_parameter = far_field_info["far_field_shape_parameter"]
        far_field_length_scale = far_field_info["far_field_length_scale"]
        far_field_blend_inner_radius = far_field_info["far_field_blend_inner_radius"]
        far_field_blend_outer_radius = far_field_info["far_field_blend_outer_radius"]
        far_field_box_min = far_field_info["far_field_box_min"]
        far_field_box_max = far_field_info["far_field_box_max"]
        far_field_rho = far_field_info["far_field_rho"]
        far_field_blend_inner_distance = far_field_info["far_field_blend_inner_distance"]
        far_field_blend_outer_distance = far_field_info["far_field_blend_outer_distance"]

        if self.global_max_centers is not None and X_global.shape[0] > int(self.global_max_centers):
            keep = structured_initial_subset_indices(X_global, int(self.global_max_centers))
            X_global = X_global[keep]
            y_global = y_global[keep]
            global_train_idx = global_train_idx[keep]

        fit_global_t0 = time.perf_counter()
        global_train_key = global_train_idx.tobytes()
        global_cache_valid = (
            self._cached_global_factor is not None
            and self._cached_c_global is not None
            and self._cached_global_train_key == global_train_key
            and self._global_tr_tree is not None
            and self._global_rho is not None
        )
        if global_cache_valid:
            global_rho = self._global_rho
            global_tr_tree = self._global_tr_tree
            global_factor = self._cached_global_factor
            c_global = self._cached_c_global
            K_global = sparse.csc_matrix((X_global.shape[0], X_global.shape[0]), dtype=float)
        elif X_global.shape[0] > 0:
            global_rho = self._compute_global_support_radius(X_global)
            global_tr_tree = cKDTree(X_global)
            global_background = self._evaluate_far_field_on_indices(
                global_train_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
            )
            y_global_residual = y_global - global_background
            K_global = assemble_wendland_matrix(
                X_global,
                X_global,
                global_rho,
                src_tree=global_tr_tree,
                chunk_size=self.assembly_chunk_size,
            ).tocsc()
            K_global = K_global + self.global_ridge * sparse.eye(K_global.shape[0], format="csc")
            global_factor = _timed_factorize_spd(
                K_global,
                timings=timings,
                key="time_global_factorization",
            )
            c_global = _timed_factor_solve(
                global_factor,
                y_global_residual,
                timings=timings,
                key="time_global_solve",
            ).reshape(-1)
            self._global_tr_tree = global_tr_tree
            self._global_rho = global_rho
            self._cached_global_factor = global_factor
            self._cached_c_global = c_global
            self._cached_global_train_key = global_train_key
            if self.freeze_global:
                self._cached_global_values = np.zeros((self.n_total,), dtype=float)
                self._cached_global_values_mask = np.zeros((self.n_total,), dtype=bool)
        else:
            global_rho = 1.0
            global_tr_tree = None
            global_factor = None
            c_global = np.zeros((0,), dtype=float)
            K_global = sparse.csc_matrix((0, 0), dtype=float)
        timings["time_fit_global"] += time.perf_counter() - fit_global_t0

        local_background = self._evaluate_background_model_on_indices(
            local_train_idx,
            X_far_field,
            c_far_field,
            far_field_shape_parameter,
            far_field_blend_inner_radius,
            far_field_blend_outer_radius,
            X_global,
            c_global,
            global_rho,
            global_tr_tree,
        )
        y_local_residual = y_local - local_background

        local_state = self._local_state
        local_update_mode = "empty"
        local_rebuild_reason = ""
        local_incremental_fallback = False
        local_stage_radius_recomputed = False

        if X_local.shape[0] > 0:
            if local_state is not None and np.array_equal(local_state.train_indices, local_train_idx):
                rho_local = np.asarray(local_state.rho_local, dtype=float).copy()
            else:
                rho_local = compute_local_support_radii(
                    X_local,
                    neighbor_k=self.neighbor_k,
                    radius_scale=self.radius_scale,
                )

            suggested_rho = (
                robust_global_support_radius(
                    rho_local,
                    quantile=self.radius_quantile,
                    inflate=self.radius_inflate,
                )
                if rho_local.size > 0
                else 1.0
            )

            needs_full_rebuild = local_state is None
            if local_state is None:
                local_rebuild_reason = "initial"
            elif local_train_idx.size < local_state.train_indices.size:
                needs_full_rebuild = True
                local_rebuild_reason = "train_set_shrank"
            elif not np.array_equal(
                local_train_idx[: min(local_train_idx.size, local_state.train_indices.size)],
                local_state.train_indices[: min(local_train_idx.size, local_state.train_indices.size)],
            ):
                needs_full_rebuild = True
                local_rebuild_reason = "non_append_update"
            elif self._force_local_rebuild_next:
                needs_full_rebuild = True
                local_rebuild_reason = "forced_rebuild"
            elif (
                self._local_radius_policy == "stage"
                and suggested_rho > (1.0 + float(self._local_radius_growth_tol)) * float(local_state.rho_global)
            ):
                needs_full_rebuild = True
                local_rebuild_reason = "radius_growth"
            elif local_state.n_updates_since_rebuild >= int(self._local_rebuild_every):
                needs_full_rebuild = True
                local_rebuild_reason = "scheduled_rebuild"

            if local_state is not None and np.array_equal(local_state.train_indices, local_train_idx) and not needs_full_rebuild:
                local_state = self._refresh_local_residual_targets(
                    local_state,
                    y_local_residual,
                    timings=timings,
                )
                local_update_mode = "reuse"
            else:
                use_incremental, incremental_reason = self._should_use_incremental_local_update(
                    local_state,
                    local_train_idx,
                    rho_local,
                )
                if not needs_full_rebuild and use_incremental:
                    try:
                        local_state = self._update_local_state_incremental(
                            state=local_state,
                            local_train_idx=local_train_idx,
                            X_local=X_local,
                            y_local_residual=y_local_residual,
                            rho_local=rho_local,
                            timings=timings,
                        )
                        local_update_mode = "incremental"
                    except Exception:
                        local_incremental_fallback = True
                        local_rebuild_reason = "incremental_fallback"
                        local_state = self._build_local_state_full(
                            local_train_idx=local_train_idx,
                            X_local=X_local,
                            y_local_residual=y_local_residual,
                            rho_local=rho_local,
                            rebuild_reason=local_rebuild_reason,
                            existing_state=local_state,
                            timings=timings,
                        )
                        local_update_mode = "full"
                        local_stage_radius_recomputed = True
                else:
                    if not needs_full_rebuild:
                        local_rebuild_reason = incremental_reason
                    local_state = self._build_local_state_full(
                        local_train_idx=local_train_idx,
                        X_local=X_local,
                        y_local_residual=y_local_residual,
                        rho_local=rho_local,
                        rebuild_reason=local_rebuild_reason,
                        existing_state=local_state,
                        timings=timings,
                    )
                    local_update_mode = "full"
                    local_stage_radius_recomputed = True
        else:
            rho_local = np.zeros((0,), dtype=float)
            local_state = None

        self._force_local_rebuild_next = False
        self._local_state = local_state
        if local_state is None:
            self._local_tr_tree = None
            self._local_rho_global = None
            local_tr_tree = None
            rho_global = 1.0
            local_factor = None
            c_local = np.zeros((0,), dtype=float)
            K_local = sparse.csc_matrix((0, 0), dtype=float)
        else:
            self._local_tr_tree = local_state.tree
            self._local_rho_global = local_state.rho_global
            local_tr_tree = local_state.tree
            rho_global = local_state.rho_global
            local_factor = local_state.factor
            c_local = np.asarray(local_state.coeffs, dtype=float).reshape(-1)
            K_local = local_state.K

        local_weights = inverse_distance_weights(
            self.all_sdf[local_check_idx],
            power=self.weight_power,
            eps=self.weight_eps,
            clip_max=1e8,
            normalize=True,
        ) if local_check_idx.size > 0 else np.zeros((0,), dtype=float)
        global_weights = inverse_distance_weights(
            self.all_sdf[global_check_idx],
            power=self.weight_power,
            eps=self.weight_eps,
            clip_max=1e8,
            normalize=True,
        ) if global_check_idx.size > 0 else np.zeros((0,), dtype=float)

        shortlist_t0 = time.perf_counter()
        X_local_check = self.all_points[local_check_idx]
        local_shortlist = self._compute_shortlist_indices(
            local_weights,
            sdf_values=self.all_sdf[local_check_idx],
            X_check=X_local_check if X_local_check.size > 0 else None,
            tr_tree=local_tr_tree,
            point_indices=local_check_idx,
            branch="local",
        )
        X_global_check = self.all_points[global_check_idx]
        global_shortlist = self._compute_shortlist_indices(
            global_weights,
            sdf_values=self.all_sdf[global_check_idx],
            X_check=X_global_check if X_global_check.size > 0 else None,
            tr_tree=global_tr_tree,
            point_indices=global_check_idx,
            branch="global",
        )
        timings["time_shortlist"] += time.perf_counter() - shortlist_t0

        if self.shortlist_diagnostics and self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            sdf_local_check = self.all_sdf[local_check_idx]
            tau0 = self.shortlist_band_zero_tol
            tau_pos = self.shortlist_band_pos_tol
            tau_neg = self.shortlist_band_neg_tol
            band_masks_check = [
                np.abs(sdf_local_check) <= tau0,
                (sdf_local_check > tau0) & (sdf_local_check <= tau_pos),
                (sdf_local_check < -tau0) & (sdf_local_check >= -tau_neg),
                sdf_local_check < -tau_neg,
            ]
            shortlist_band_counts_check = [int(np.sum(m)) for m in band_masks_check]
            if local_shortlist.size > 0:
                sdf_local_short = self.all_sdf[local_check_idx[local_shortlist]]
                shortlist_band_counts_short = [
                    int(np.sum(np.abs(sdf_local_short) <= tau0)),
                    int(np.sum((sdf_local_short > tau0) & (sdf_local_short <= tau_pos))),
                    int(np.sum((sdf_local_short < -tau0) & (sdf_local_short >= -tau_neg))),
                    int(np.sum(sdf_local_short < -tau_neg)),
                ]
            else:
                shortlist_band_counts_short = [0, 0, 0, 0]
        else:
            shortlist_band_counts_check = [0, 0, 0, 0]
            shortlist_band_counts_short = [0, 0, 0, 0]

        A_l_lc = None
        shortlist_pred_t0 = time.perf_counter()
        if local_shortlist.size > 0:
            lc_short_global_idx = local_check_idx[local_shortlist]
            background_pred_lc_short = self._evaluate_background_model_on_indices(
                lc_short_global_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )
            X_lc_short = self.all_points[lc_short_global_idx]
            if X_local.shape[0] > 0:
                if self.enable_adjoint:
                    A_l_lc = self._evaluate_local_operator(
                        X_local,
                        X_lc_short,
                        rho_global,
                        local_tr_tree,
                    )
                    local_pred_lc_short = np.asarray(A_l_lc @ c_local, dtype=float).reshape(-1)
                else:
                    local_pred_lc_short = self._evaluate_local_values(
                        X_lc_short,
                        X_local,
                        c_local,
                        rho_global,
                        local_tr_tree,
                    )
            else:
                local_pred_lc_short = np.zeros((local_shortlist.size,), dtype=float)
            pred_lc_short = background_pred_lc_short + local_pred_lc_short
            r_lc_short = pred_lc_short - self.all_sdf[lc_short_global_idx]
        else:
            lc_short_global_idx = np.zeros((0,), dtype=int)
            pred_lc_short = np.zeros((0,), dtype=float)
            r_lc_short = np.zeros((0,), dtype=float)

        if global_shortlist.size > 0:
            gc_short_global_idx = global_check_idx[global_shortlist]
            background_pred_gc_short = self._evaluate_background_model_on_indices(
                gc_short_global_idx,
                X_far_field,
                c_far_field,
                far_field_shape_parameter,
                far_field_blend_inner_radius,
                far_field_blend_outer_radius,
                X_global,
                c_global,
                global_rho,
                global_tr_tree,
            )
            X_gc_short = self.all_points[gc_short_global_idx]
            if X_local.shape[0] > 0:
                local_pred_gc_short = self._evaluate_local_values(
                    X_gc_short,
                    X_local,
                    c_local,
                    rho_global,
                    local_tr_tree,
                )
            else:
                local_pred_gc_short = np.zeros((global_shortlist.size,), dtype=float)
            pred_gc_short = background_pred_gc_short + local_pred_gc_short
            r_gc_short = pred_gc_short - self.all_sdf[gc_short_global_idx]
        else:
            r_gc_short = np.zeros((0,), dtype=float)
        timings["time_shortlist_prediction"] += time.perf_counter() - shortlist_pred_t0

        local_residual = np.zeros((local_check_idx.size,), dtype=float)
        local_residual[local_shortlist] = r_lc_short
        global_residual = np.zeros((global_check_idx.size,), dtype=float)
        global_residual[global_shortlist] = r_gc_short

        adjoint_t0 = time.perf_counter()
        if self.enable_adjoint and local_factor is not None and A_l_lc is not None and r_lc_short.size > 0:
            w_lc_short = local_weights[local_shortlist]
            adjoint_rhs = np.asarray(A_l_lc.T @ (w_lc_short * r_lc_short), dtype=float).reshape(-1)
            lambda_local = _timed_factor_solve(
                local_factor,
                adjoint_rhs,
                timings=timings,
                key="time_adjoint_solve",
            ).reshape(-1)
            psi_lc_short = np.asarray(A_l_lc @ lambda_local, dtype=float).reshape(-1)
            eta_lc_short = np.abs(r_lc_short) * np.abs(psi_lc_short)
        else:
            lambda_local = np.zeros((X_local.shape[0],), dtype=float)
            psi_lc_short = np.zeros((local_shortlist.size,), dtype=float)
            eta_lc_short = (
                np.abs(r_lc_short) * local_weights[local_shortlist]
                if local_shortlist.size > 0
                else np.zeros((0,), dtype=float)
            )
        timings["time_adjoint"] += time.perf_counter() - adjoint_t0

        local_eta = np.zeros((local_check_idx.size,), dtype=float)
        local_eta[local_shortlist] = self.local_error_weight * eta_lc_short
        local_selection_score = np.zeros((local_check_idx.size,), dtype=float)
        local_selection_terms_short = {
            "feature_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "curvature_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "sign_bonus": np.zeros((local_shortlist.size,), dtype=float),
            "surface_proximity": np.zeros((local_shortlist.size,), dtype=float),
        }
        if local_shortlist.size > 0:
            selection_short, local_selection_terms_short = self._compute_local_selection_scores(
                point_indices=lc_short_global_idx,
                sdf_values=self.all_sdf[lc_short_global_idx],
                pred_values=pred_lc_short,
                eta_base=self.local_error_weight * eta_lc_short,
            )
            local_selection_score[local_shortlist] = selection_short
        global_eta = np.zeros_like(global_residual)

        info = {
            "X_far_field": X_far_field,
            "y_far_field": y_far_field,
            "far_field_center": far_field_center,
            "far_field_reference_radius": far_field_reference_radius,
            "far_field_box_min": far_field_box_min,
            "far_field_box_max": far_field_box_max,
            "local_train_idx": local_train_idx,
            "global_train_idx": global_train_idx,
            "local_check_idx": local_check_idx,
            "global_check_idx": global_check_idx,
            "X_local": X_local,
            "X_global": X_global,
            "c_far_field": c_far_field,
            "y_local": y_local,
            "y_global": y_global,
            "c_local": c_local,
            "c_global": c_global,
            "K_far_field": K_far_field,
            "K_local": K_local,
            "K_global": K_global,
            "far_field_factor": far_field_factor,
            "local_factor": local_factor,
            "global_factor": global_factor,
            "far_field_shape_parameter": far_field_shape_parameter,
            "far_field_length_scale": far_field_length_scale,
            "far_field_blend_inner_radius": far_field_blend_inner_radius,
            "far_field_blend_outer_radius": far_field_blend_outer_radius,
            "far_field_rho": far_field_rho,
            "far_field_blend_inner_distance": far_field_blend_inner_distance,
            "far_field_blend_outer_distance": far_field_blend_outer_distance,
            "rho_local": rho_local,
            "rho_global": rho_global,
            "global_rho": global_rho,
            "local_tr_tree": local_tr_tree,
            "global_tr_tree": global_tr_tree,
            "local_residual": local_residual,
            "global_residual": global_residual,
            "local_weights": local_weights,
            "global_weights": global_weights,
            "local_eta": local_eta,
            "global_eta": global_eta,
            "local_selection_score": local_selection_score,
            "local_shortlist": local_shortlist,
            "global_shortlist": global_shortlist,
            "shortlist_band_counts_check": shortlist_band_counts_check,
            "shortlist_band_counts_shortlist": shortlist_band_counts_short,
            "lambda_local": lambda_local,
            "psi_local_short": psi_lc_short,
            "local_selection_terms_short": local_selection_terms_short,
            "local_oml_max_abs_error": 0.0,
            "timings": timings,
            "local_update_mode": local_update_mode,
            "local_rebuild_reason": local_rebuild_reason,
            "local_stage_radius_recomputed": local_stage_radius_recomputed,
            "local_incremental_fallback": local_incremental_fallback,
            "local_n_rebuilds": 0 if local_state is None else int(local_state.n_rebuilds),
            "local_n_updates_since_rebuild": 0 if local_state is None else int(local_state.n_updates_since_rebuild),
        }

        if self.shortlist_diagnostics:
            oml_idx = np.asarray(self.oml_point_indices, dtype=int)
            if oml_idx.size > 0:
                oml_eval_t0 = time.perf_counter()
                pred_oml = np.zeros((oml_idx.size,), dtype=float)
                if X_far_field.shape[0] > 0:
                    pred_oml += self._evaluate_far_field_values(
                        self.all_points[oml_idx],
                        X_far_field,
                        c_far_field,
                        far_field_shape_parameter,
                        far_field_blend_inner_radius,
                        far_field_blend_outer_radius,
                    )
                if X_global.shape[0] > 0:
                    pred_oml += self._evaluate_global_model_on_indices(
                        oml_idx,
                        X_global,
                        c_global,
                        global_rho,
                        global_tr_tree,
                    )
                if X_local.shape[0] > 0:
                    pred_oml += self._evaluate_local_values(
                        self.all_points[oml_idx],
                        X_local,
                        c_local,
                        rho_global,
                        local_tr_tree,
                    )
                oml_eval_time = time.perf_counter() - oml_eval_t0
                oml_abs_error = np.abs(pred_oml - self.all_sdf[oml_idx])
                oml_l2 = float(np.sqrt(np.sum(oml_abs_error ** 2)))
                info["local_oml_max_abs_error"] = float(np.max(oml_abs_error)) if oml_abs_error.size > 0 else 0.0
                timings["time_oml_surrogate_eval_total"] = oml_eval_time
                timings["time_oml_surrogate_eval_per_point"] = oml_eval_time / float(oml_idx.size)
            else:
                oml_l2 = 0.0

            train_idx = np.unique(np.concatenate([local_train_idx, global_train_idx])).astype(int)
            if train_idx.size > 0:
                pred_train = np.zeros((train_idx.size,), dtype=float)
                if X_far_field.shape[0] > 0:
                    pred_train += self._evaluate_far_field_values(
                        self.all_points[train_idx],
                        X_far_field,
                        c_far_field,
                        far_field_shape_parameter,
                        far_field_blend_inner_radius,
                        far_field_blend_outer_radius,
                    )
                if X_global.shape[0] > 0:
                    pred_train += self._evaluate_global_model_on_indices(
                        train_idx,
                        X_global,
                        c_global,
                        global_rho,
                        global_tr_tree,
                    )
                if X_local.shape[0] > 0:
                    pred_train += self._evaluate_local_values(
                        self.all_points[train_idx],
                        X_local,
                        c_local,
                        rho_global,
                        local_tr_tree,
                    )
                train_l2 = float(np.sqrt(np.sum((pred_train - self.all_sdf[train_idx]) ** 2)))
            else:
                train_l2 = 0.0
            info["oml_l2_error"] = oml_l2
            info["train_l2_error"] = train_l2

        timings["time_total_fit"] = time.perf_counter() - total_t0
        self._last_fit_info = info
        return info

    def _build_diagnostics_from_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        _local_sdf = self.all_sdf[info["local_train_idx"]]
        diagnostics = {
            "n_far_field_train": int(info["X_far_field"].shape[0]),
            "n_local_train": int(info["local_train_idx"].size),
            "n_global_train": int(info["global_train_idx"].size),
            "n_local_train_oml": int(np.sum(np.isclose(_local_sdf, 0.0))),
            "n_local_train_positive": int(np.sum(_local_sdf > 0.0)),
            "n_local_train_negative": int(np.sum(_local_sdf < 0.0)),
            "n_local_check": int(info["local_check_idx"].size),
            "n_global_check": int(info["global_check_idx"].size),
            "far_field_shape_parameter": float(info["far_field_shape_parameter"]),
            "far_field_length_scale": float(info["far_field_length_scale"]),
            "far_field_reference_radius": float(info["far_field_reference_radius"]),
            "far_field_blend_inner_radius": float(info["far_field_blend_inner_radius"]),
            "far_field_blend_outer_radius": float(info["far_field_blend_outer_radius"]),
            "far_field_rho": float(info.get("far_field_rho", info["far_field_shape_parameter"])),
            "far_field_blend_inner_distance": float(
                info.get("far_field_blend_inner_distance", info["far_field_blend_inner_radius"])
            ),
            "far_field_blend_outer_distance": float(
                info.get("far_field_blend_outer_distance", info["far_field_blend_outer_radius"])
            ),
            "rho_global": float(info["rho_global"]),
            "global_rho": float(info["global_rho"]),
            "residual_l2_weighted_local": float(
                np.sqrt(np.sum(info["local_weights"] * info["local_residual"] ** 2))
            ) if info["local_residual"].size > 0 else 0.0,
            "residual_l2_weighted_global": float(
                np.sqrt(np.sum(info["global_weights"] * info["global_residual"] ** 2))
            ) if info["global_residual"].size > 0 else 0.0,
            "eta_max_local": float(np.max(info["local_eta"])) if info["local_eta"].size > 0 else 0.0,
            "eta_max_global": float(np.max(info["global_eta"])) if info["global_eta"].size > 0 else 0.0,
            "selection_score_max_local": float(np.max(info["local_selection_score"]))
            if info["local_selection_score"].size > 0
            else 0.0,
            "local_oml_max_abs_error": float(info["local_oml_max_abs_error"]),
            "adjoint_enabled": bool(self.enable_adjoint),
            "selection_score_mode": self.selection_score_mode,
            "geometry_metadata_enabled": bool(self.has_geometry_metadata),
            "hybrid_mode": (
                "sequential_far_field_plus_background_plus_local_optimized"
                if info["X_far_field"].shape[0] > 0
                else "sequential_background_plus_local_optimized"
            ),
            "local_solver_mode": "incremental",
            "local_update_mode": info.get("local_update_mode", "full"),
            "local_rebuild_reason": info.get("local_rebuild_reason", ""),
            "local_stage_radius_recomputed": bool(info.get("local_stage_radius_recomputed", False)),
            "local_incremental_fallback": bool(info.get("local_incremental_fallback", False)),
            "local_incremental_used": info.get("local_update_mode", "") == "incremental",
            "local_n_rebuilds": int(info.get("local_n_rebuilds", 0)),
            "local_n_updates_since_rebuild": int(info.get("local_n_updates_since_rebuild", 0)),
        }
        diagnostics.update(info.get("timings", {}))
        if "oml_l2_error" in info:
            diagnostics["oml_l2_error"] = float(info["oml_l2_error"])
        if "train_l2_error" in info:
            diagnostics["train_l2_error"] = float(info["train_l2_error"])
        return diagnostics

    def _build_result_from_info(
        self,
        info: Dict[str, Any],
        *,
        iteration: int,
        selected_local_global: np.ndarray,
        selected_global_global: np.ndarray,
        min_separation_local: Optional[float],
        min_separation_global: Optional[float],
        refit_after_update: Optional[bool],
    ) -> DWRRefinementResult:
        residual = np.zeros((self.n_total,), dtype=float)
        weights = np.zeros((self.n_total,), dtype=float)
        eta = np.zeros((self.n_total,), dtype=float)
        psi = np.zeros((self.n_total,), dtype=float)

        residual[info["local_check_idx"]] = info["local_residual"]
        residual[info["global_check_idx"]] = info["global_residual"]
        weights[info["local_check_idx"]] = info["local_weights"]
        weights[info["global_check_idx"]] = info["global_weights"]
        eta[info["local_check_idx"]] = info["local_eta"]
        eta[info["global_check_idx"]] = info["global_eta"]
        if info["local_shortlist"].size > 0:
            psi[info["local_check_idx"][info["local_shortlist"]]] = info["psi_local_short"]

        diagnostics = self._build_diagnostics_from_info(info)
        if refit_after_update is not None:
            diagnostics.update(
                {
                    "iteration": int(iteration),
                    "n_added_local": int(selected_local_global.size),
                    "n_added_global": int(selected_global_global.size),
                    "min_separation_local": float(min_separation_local),
                    "min_separation_global": float(min_separation_global),
                    "refit_after_update": bool(refit_after_update),
                    "shortlist_band_counts_check": info.get("shortlist_band_counts_check", [0, 0, 0, 0]),
                    "shortlist_band_counts_shortlist": info.get("shortlist_band_counts_shortlist", [0, 0, 0, 0]),
                }
            )

        return DWRRefinementResult(
            iteration=int(iteration),
            train_indices=np.unique(
                np.concatenate([info["local_train_idx"], info["global_train_idx"]])
            ).astype(int),
            check_indices=np.unique(
                np.concatenate([info["local_check_idx"], info["global_check_idx"]])
            ).astype(int),
            rho_local=info["rho_local"],
            rho_global=info["rho_global"],
            residual=residual,
            weights=weights,
            lambda_vec=info["lambda_local"],
            psi=psi,
            eta=eta,
            selected_check_indices=np.concatenate([selected_local_global, selected_global_global]).astype(int),
            selected_global_indices=np.concatenate([selected_local_global, selected_global_global]).astype(int),
            diagnostics=diagnostics,
        )

    def compute_dwr_scores(self) -> DWRRefinementResult:
        info = self._fit_current_model()
        return self._build_result_from_info(
            info,
            iteration=-1,
            selected_local_global=np.array([], dtype=int),
            selected_global_global=np.array([], dtype=int),
            min_separation_local=None,
            min_separation_global=None,
            refit_after_update=None,
        )

    def refine_once(
        self,
        n_add_local: int,
        n_add_global: int = 0,
        min_separation_local: Optional[float] = None,
        min_separation_global: Optional[float] = None,
        iteration: int = -1,
        recompute_scores_after_update: bool = False,
    ) -> DWRRefinementResult:
        info = self._fit_current_model()

        if min_separation_local is None:
            min_separation_local = 0.5 * info["rho_global"]
        if min_separation_global is None:
            min_separation_global = 0.5 * info["global_rho"]

        local_shortlist = info["local_shortlist"]
        selected_local_local = greedy_farthest_subset_from_scores(
            self.all_points[info["local_check_idx"]][local_shortlist],
            info["local_selection_score"][local_shortlist],
            n_select=int(n_add_local),
            min_separation=float(min_separation_local),
        ) if local_shortlist.size > 0 and int(n_add_local) > 0 else np.array([], dtype=int)
        selected_local_global = (
            info["local_check_idx"][local_shortlist[selected_local_local]]
            if selected_local_local.size > 0
            else np.array([], dtype=int)
        )
        selected_local_global = self._append_local_train_indices(selected_local_global)

        if self.shortlist_diagnostics and self.shortlist_mode in {"banded_weight", "geometry_banded_weight"}:
            selected_band_counts = [0, 0, 0, 0]
            if selected_local_global.size > 0:
                sdf_sel = self.all_sdf[selected_local_global]
                tau0 = self.shortlist_band_zero_tol
                tau_pos = self.shortlist_band_pos_tol
                tau_neg = self.shortlist_band_neg_tol
                selected_band_counts = [
                    int(np.sum(np.abs(sdf_sel) <= tau0)),
                    int(np.sum((sdf_sel > tau0) & (sdf_sel <= tau_pos))),
                    int(np.sum((sdf_sel < -tau0) & (sdf_sel >= -tau_neg))),
                    int(np.sum(sdf_sel < -tau_neg)),
                ]
        else:
            selected_band_counts = [0, 0, 0, 0]

        if not self.freeze_global and int(n_add_global) > 0:
            global_shortlist = info["global_shortlist"]
            selected_global_local = greedy_farthest_subset_from_scores(
                self.all_points[info["global_check_idx"]][global_shortlist],
                info["global_eta"][global_shortlist],
                n_select=int(n_add_global),
                min_separation=float(min_separation_global),
            ) if global_shortlist.size > 0 else np.array([], dtype=int)
            selected_global_global = (
                info["global_check_idx"][global_shortlist[selected_global_local]]
                if selected_global_local.size > 0
                else np.array([], dtype=int)
            )
            if selected_global_global.size > 0:
                self.global_train_mask[selected_global_global] = True
                self._invalidate_global_cache()
        else:
            selected_global_global = np.array([], dtype=int)

        if recompute_scores_after_update:
            res = self.compute_dwr_scores()
            res.diagnostics.update(
                {
                    "iteration": int(iteration),
                    "n_added_local": int(selected_local_global.size),
                    "n_added_global": int(selected_global_global.size),
                    "min_separation_local": float(min_separation_local),
                    "min_separation_global": float(min_separation_global),
                    "refit_after_update": True,
                    "shortlist_band_counts_check": info.get("shortlist_band_counts_check", [0, 0, 0, 0]),
                    "shortlist_band_counts_shortlist": info.get("shortlist_band_counts_shortlist", [0, 0, 0, 0]),
                    "selected_band_counts": selected_band_counts,
                    "oml_l2_error": info.get("oml_l2_error", 0.0),
                    "train_l2_error": info.get("train_l2_error", 0.0),
                }
            )
            res.selected_check_indices = np.concatenate([selected_local_global, selected_global_global]).astype(int)
            res.selected_global_indices = res.selected_check_indices.copy()
            res.iteration = int(iteration)
            return res

        res = self._build_result_from_info(
            info,
            iteration=iteration,
            selected_local_global=selected_local_global,
            selected_global_global=selected_global_global,
            min_separation_local=min_separation_local,
            min_separation_global=min_separation_global,
            refit_after_update=False,
        )
        res.diagnostics.update(
            {
                "selected_band_counts": selected_band_counts,
                "oml_l2_error": info.get("oml_l2_error", 0.0),
                "train_l2_error": info.get("train_l2_error", 0.0),
            }
        )
        return res



# ============================================================
# Optional: exact full-Delta-J reranking on a shortlist
# ============================================================

def full_delta_j_shortlist(
    train_points: np.ndarray,
    train_sdf: np.ndarray,
    coeffs: np.ndarray,
    factor,
    check_points: np.ndarray,
    check_sdf: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    rho_global: float,
    shortlist_points: np.ndarray,
    shortlist_sdf: np.ndarray,
) -> np.ndarray:
    """
    Compute exact one-center predicted Delta J on a shortlist of candidates,
    using the interpolation Schur-complement formula.

    This is intended for late-stage reranking only.

    Parameters
    ----------
    train_points, train_sdf : current training set
    coeffs : current interpolation coefficients c
    factor : CHOLMOD factorization for K
    check_points, check_sdf : current check set
    residual : current check residual r = Bc - d_check
    weights : inverse-distance weights on check set
    rho_global : global support radius used for K and B
    shortlist_points, shortlist_sdf : candidate points to rerank

    Returns
    -------
    delta_j : (M,) ndarray
        Exact one-step predicted Delta J for each shortlist candidate.
        More negative is better.
    """
    tr_tree = cKDTree(train_points)
    chk_tree = cKDTree(check_points)

    # Precompute B once if you do many calls externally; kept inline here for simplicity
    B = assemble_wendland_matrix(train_points, check_points, rho_global, src_tree=tr_tree).tocsr()

    out = np.zeros(shortlist_points.shape[0], dtype=float)
    kappa = wendland_c2(np.array([0.0]))[0]  # phi(0) = 1 for Wendland C2

    for m, (x_star, b_star) in enumerate(zip(shortlist_points, shortlist_sdf)):
        # k: coupling to current training set
        idx_tr = tr_tree.query_ball_point(x_star, r=rho_global)
        k = np.zeros(train_points.shape[0], dtype=float)
        if idx_tr:
            idx_tr = np.asarray(idx_tr, dtype=int)
            r_tr = np.linalg.norm(train_points[idx_tr] - x_star[None, :], axis=1)
            k[idx_tr] = wendland_c2(r_tr / rho_global)

        # Solve u = K^{-1} k
        u = factor(k)

        pred_star = float(k @ coeffs)
        num = b_star - pred_star
        den = kappa - float(k @ u)

        # Skip near-redundant / unstable candidates
        if den <= 1e-14:
            out[m] = np.inf
            continue

        alpha = num / den

        # g: new basis evaluated at check points
        idx_chk = chk_tree.query_ball_point(x_star, r=rho_global)
        g = np.zeros(check_points.shape[0], dtype=float)
        if idx_chk:
            idx_chk = np.asarray(idx_chk, dtype=int)
            r_chk = np.linalg.norm(check_points[idx_chk] - x_star[None, :], axis=1)
            g[idx_chk] = wendland_c2(r_chk / rho_global)

        p = g - B @ u

        delta_j = alpha * np.dot(p, weights * residual) + 0.5 * alpha**2 * np.dot(p, weights * p)
        out[m] = delta_j

    return out


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Example placeholders:
    # Replace these with your own dense labeled SDF dataset.
    # --------------------------------------------------------
    rng = np.random.default_rng(3)

    # Example cloud in a box, with fake signed distances to a sphere
    n_total = 5000
    pts = rng.uniform(-1.5, 1.5, size=(n_total, 3))
    sdf = np.linalg.norm(pts, axis=1) - 1.0

    # Initial foreground training subset:
    # here just random as a placeholder
    initial_train = rng.choice(n_total, size=350, replace=False)

    refiner = WendlandC2DWRRefiner(
        all_points=pts,
        all_sdf=sdf,
        initial_train_indices=initial_train,
        neighbor_k=24,
        radius_scale=1.6,
        radius_quantile=0.90,
        radius_inflate=1.10,
        weight_power=1.0,
        weight_eps=1e-3,
        jitter=1e-10,
    )

    # Inspect current scores
    res0 = refiner.compute_dwr_scores()
    print("Initial weighted residual norm:", res0.diagnostics["residual_l2_weighted"])
    print("Initial global support radius:", res0.rho_global)

    # Run a few refinement steps
    history = refiner.refine(
        n_iterations=5,
        n_add_per_iter=60,
        min_separation=None,
        verbose=True,
    )
