import scipy.sparse.linalg as sla
from utils.geometry_util import hash_arrays, torch2np


import numpy as np
import scipy.sparse
import torch


import os
import os.path as osp
import igl
try:
    import pyshell
except ImportError:
    print("Warning: Failed to import pyshell. Some functionality may be limited.")
from utils.cache_util import get_cached_compute

def computeEV(v, f, k, bending_weight=1e-2, sigma=None, which=None, nonzero=True):
    """
    Compute eigenvectors of shell hessian 
        Args:
            fix (optional): vertex indices for boundary values

        returns:
            vals : eigenvalues
            vecs: eigenvectors 3n x k 
    """

    f = f.astype(np.int32)
    # edge flaps
    uE, EMAP, EF, EI = igl.edge_flaps(f)
    # massmatrix
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    M = scipy.sparse.block_diag((M, M, M), format='lil')
    k = k + 6

    hess = pyshell.shell_deformed_hessian(v, v, f, uE, EMAP, EF, EI, bending_weight)

    # hessian is quite poorly conditioned
    eps = 1e-6
    hess = (hess + eps * scipy.sparse.identity(hess.shape[0])).tocsc()

    if sigma is None:
        if which is None:
            # note: sigma is no longer equal to 0 like in original implementation
            sigma = eps
            which = 'LM'
            vals, vecs = sla.eigsh(hess, k, M=M, sigma=0, which=which)
            # vals_eps, vecs_eps = sla.eigsh(hess_eps, k, M=M, sigma=sigma, which=which)
        else:
            vals, vecs = sla.eigsh(hess, k, M=M, which=which)
    else:
        vals, vecs = sla.eigsh(hess, k, M=M, sigma=sigma)

    if nonzero:  # remove vecs corresponding to zero vals (six dimensional kernel due to rigid body motions)
        ind = vals > 1e-8

        if np.sum(ind) < k - 6:
            print('mesh has probably irregularities, found ' + str(k - 6 - np.sum(ind)) + 'many EF with zero EV')
            # vals,vecs =  linalg.eigsh(hess, k+np.sum(ind)-6,M=M,sigma = sigma,which = which)
            ind = vals > 1e-8

        vals = vals[6:]
        vecs = vecs[:, 6:]

    # fix signs
    ind = vecs[0, :] < 0
    vals[ind] *= -1
    vecs[:, ind] *= -1

    return vals, vecs


def computeVectorBasis(v, f, k=200, bending_weight=1e-2, nonzero=True):
    """
        computes elastic vibration modes (i.e. vector valued eigenfunctions of elastic hessian)
        args:
            k: first k eigennfunctions (lowest eigenvalues)
            bending_weight: weighting term of membrane and bending energy 
            nonzero: (boolean) compute eigenfunctions with nonzero eigenvalue (kernel with r.b.m. is removed)
    """
    # print("Computing " + str(k) + ' elastic eigenfunctions for ' + self.name)
    Evalues, vectorBasis = computeEV(v, f, k, bending_weight=bending_weight,
                                                        nonzero=nonzero)
    return Evalues, vectorBasis


def projection(normals):
    """
    Create a projection matrix for projection on vertex normals
        args:
            normals: m x 3

        returns:
            P: projection matrix 3*m x m 

    """
    # normals in m x 3
    m = normals.shape[0]
    P = scipy.sparse.lil_matrix((3 * m, m), dtype='d')
    for j in range(m):
        P[j, j] = normals[j, 0]
        P[j + m, j] = normals[j, 1]
        P[j + 2 * m, j] = normals[j, 2]
    return P


def computeElasticBasis(verts, faces, k=200, bending_weight=1e-2, nonzero=True, rescale_unit_area=False):
    v, f = verts, faces

    if rescale_unit_area:
            area = np.sqrt(np.sum(igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)))
            v = (v / area)
    f = f.astype(np.int32)
    nVert = v.shape[0]
    mass = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    normals = igl.per_vertex_normals(v, f)
    adj = igl.adjacency_list(f)

    Evalues, vectorBasis = computeVectorBasis(v, f, k=k, bending_weight=bending_weight, nonzero=nonzero)
    elasticBasis = projection(normals).T.dot(vectorBasis)

    elasticBasis = harmonic_inpaint_nans(v, f, elasticBasis)
    # convert to a vector instead of a sparse matrix
    mass = mass.diagonal()
    return mass, Evalues, elasticBasis


def get_elas_operators(verts, faces, k=200, bending_weight=1e-2,
                  cache_dir=None):
    return get_cached_compute(computeElasticBasis, verts=verts, faces=faces, k=k, bending_weight=bending_weight, cache_dir=cache_dir)

def projectOnNormals(v, f, function):
    """
    project a 3n function on vertex normals
        args:
            v, f: mesh
            function: n x 3
        returns:
            proj : scalar valued function on mesh
    """

    normals = igl.per_vertex_normals(v, f)

    proj = [normals[i] * np.dot(function[i], normals[i]) for i in range(normals.shape[0])]

    return proj



def harmonic_inpaint_nans(V, F, values, eps=0.0, use_factor=True):
    """
    Fill NaNs in `values` (N x D) by harmonic interpolation on mesh (V,F),
    assuming NaNs occur on the SAME ROWS for all channels.

    Args:
        V: (N,3) vertices
        F: (M,3) faces (int)
        values: (N,D) array with NaNs on identical rows across columns
        eps: small diagonal regularizer for A (0.0 disables)
        use_factor: if True, factorize once (faster for many D)

    Returns:
        (N,D) array with NaNs filled.
    """
    N = V.shape[0]
    D = values.shape[1] if values.ndim > 1 else 1
    vals = values if values.ndim == 2 else values[:, None]

    # Early exit if nothing to fill
    nan_mask_rows = np.isnan(vals).any(axis=1)
    if not nan_mask_rows.any():
        return values.copy()

    print(f"Warning: There are {len(nan_mask_rows.nonzero())} NaN rows in the elastic basis, filling them with harmonic interpolation")
    # Safety: assert the "shared rows" assumption
    if not np.all(nan_mask_rows == np.isnan(vals[:, 0])):
        raise ValueError("NaNs are not on identical rows across channels.")

    # Build Laplacian
    L = igl.cotmatrix(V, F)  # (N,N) sparse, negative semidefinite

    b_idx = np.flatnonzero(~nan_mask_rows)  # boundary (known)
    i_idx = np.flatnonzero(nan_mask_rows)   # interior (unknown)

    if b_idx.size == 0:
        raise ValueError("No boundary (known) rows; cannot inpaint.")
    if i_idx.size == 0:
        return values.copy()  # everything known already

    # Blocks
    A = L[i_idx][:, i_idx].tocsr()
    B = L[i_idx][:, b_idx].tocsr()

    if eps > 0.0:
        A = A + eps * scipy.sparse.eye(A.shape[0], format='csr')

    # RHS for all channels at once: A U_i = -B U_b
    U_b = vals[b_idx, :]                         # (|B|, D)
    RHS = -B @ U_b                               # (|I|, D) dense

    # Solve once for all RHS
    if use_factor:
        # Factorize once and solve (faster for multiple columns)
        A_csc = A.tocsc()
        lu = sla.splu(A_csc)
        U_i = lu.solve(RHS.astype(np.float32))   # (|I|, D)
    else:
        # Direct multi-RHS solve (factorization not explicitly reused)
        U_i = sla.spsolve(A, RHS.astype(np.float32))  # (|I|, D)

    # Assemble output
    out = vals.copy()
    out[i_idx, :] = U_i
    return out if values.ndim == 2 else out[:, 0]