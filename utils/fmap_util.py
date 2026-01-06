import numpy as np
import torch
from utils.tensor_util import to_numpy
import scipy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def spectral_mass_computation(elas_evecs, mass):
    M = torch.diag(mass)
    Mk = elas_evecs.t() @ M @ elas_evecs
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness
    sqrtMk = torch.tensor(sqrtMk).to(elas_evecs.device).float()
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    return Mk, sqrtMk, invsqrtMk

def nn_query(feat_x, feat_y, dim=-2):
    """
    Find correspondences via nearest neighbor query
    Args:
        feat_x: feature vector of shape x. [V1, C].
        feat_y: feature vector of shape y. [V2, C].
        dim: number of dimension
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2].
    """
    dist = torch.cdist(feat_x, feat_y)  # [V1, V2]
    p2p = dist.argmin(dim=dim)
    return p2p

def pointmap2fmap(p2p, evecs_x, evecs_y):
    """
    Compute functional map from point-to-point map
    Arg:
        p2p: point-to-point map (shape y -> shape x). [V2]
    Return:
        Cxy: functional map (shape x -> shape y). Shape [K, K]
    """
    evecs_x_a = evecs_x[p2p]
    evecs_y_a = evecs_y

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def elas_pointmap2fmap(p2p, evecs_x, evecs_y, mass_x, mass_y):
    """
    Compute general(elastic) functional map from point-to-point map
    Args:
        p2p: point-to-point map (shape y -> shape x). [V2]
    Returns:
        Cxy: functional map (shape x -> shape y). Shape [K, K]
    """
    evecs_x_a = evecs_x[p2p]
    evecs_y_a = evecs_y
    mass_y_a = mass_y
    sqrt_mass_y_a = torch.sqrt(mass_y_a)
    evecs_x_a = torch.diag(sqrt_mass_y_a) @ evecs_x_a
    evecs_y_a = torch.diag(sqrt_mass_y_a) @ evecs_y_a

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def _fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    helper function to convert functional map to point-to-point map
    """
    dataA = evecs_x @ Cxy.t()
    dataB = evecs_y

    return dataA, dataB

def fmap2pointmap(Cxy, evecs_x, evecs_y):
    """
    Convert functional map to point-to-point map

    Args:
        Cxy: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dataA, dataB = _fmap2pointmap(Cxy, evecs_x, evecs_y)
    p2p = nn_query(dataA, dataB)
    return p2p

def _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    helper function to convert general(elastic) functional map to point-to-point map
    """
    M1k, sqrtM1k, invsqrtM1k = spectral_mass_computation(elas_evecs_x, mass1)
    M2k, sqrtM2k, invsqrtM2k = spectral_mass_computation(elas_evecs_y, mass2)

    dataA = elas_evecs_x @ torch.inverse(M1k) @ elas_Cxy.t() @ M2k @ invsqrtM2k
    dataB = elas_evecs_y @ invsqrtM2k

    return dataA, dataB

def elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    Convert general(elastic) functional map to point-to-point map
    
    Args:
        elas_Cxy: general(elastic) functional map (shape x->shape y). Shape [K, K]
        elas_evecs_x: eigenvectors of shape x. Shape [V1, K]
        elas_evecs_y: eigenvectors of shape y. Shape [V2, K]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    dataA, dataB = _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2)
    return nn_query(dataA, dataB)

def hybrid_fmap2pointmap(Cxy, evecs_x, evecs_y, elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2):
    """
    Convert hybrid functional map to point-to-point map
    
    Args:
        Cxy: functional map (shape x->shape y). Shape [K, K]
        evecs_x: eigenvectors of shape x. Shape [V1, K]
        evecs_y: eigenvectors of shape y. Shape [V2, K]
        elas_Cxy: general(elastic) functional map (shape x->shape y). Shape [K, K]
        elas_evecs_x: eigenvectors of shape x. Shape [V1, K]
        elas_evecs_y: eigenvectors of shape y. Shape [V2, K]
        mass1: mass of shape x. Shape [V1]
        mass2: mass of shape y. Shape [V2]
    Returns:
        p2p: point-to-point map (shape y -> shape x). [V2]
    """
    lb_dataA, lb_dataB = _fmap2pointmap(Cxy, evecs_x, evecs_y)
    elas_dataA, elas_dataB = _elas_fmap2pointmap(elas_Cxy, elas_evecs_x, elas_evecs_y, mass1, mass2)

    # merge embeddings
    merged_dataA = torch.cat((elas_dataA, lb_dataA), dim=1)
    merged_dataB = torch.cat((elas_dataB, lb_dataB), dim=1)

    p2p = nn_query(merged_dataA, merged_dataB)

    return p2p

def zoomout(p2p, evecs_x, evecs_y, k_init=30, k_final=200):
    assert evecs_x.size(1) >= k_final

    for k in tqdm(range(k_init, k_final+1)):
        Cxy = pointmap2fmap(p2p, evecs_x[:, :k], evecs_y[:, :k])
        p2p = fmap2pointmap(Cxy, evecs_x[:, :k], evecs_y[:, :k])
    
    return p2p

def elas_zoomout(p2p, evecs_x, evecs_y, mass1, mass2, k_init=30, k_final=200):
    assert evecs_x.size(1) >= k_final

    for k in tqdm(range(k_init, k_final+1)):
        Cxy = elas_pointmap2fmap(p2p, evecs_x[:, :k], evecs_y[:, :k], mass1, mass2)
        p2p = elas_fmap2pointmap(Cxy, evecs_x[:, :k], evecs_y[:, :k], mass1, mass2)
    
    return p2p

def hybrid_zoomout(p2p, evecs_x, evecs_y, elas_evecs_x, elas_evecs_y, mass1, mass2, k_init=(20,10), k_final=(100,100)):
    assert evecs_x.size(1) >= k_final[0]
    assert elas_evecs_x.size(1) >= k_final[1]

    start_lb, start_elas = k_init
    end_lb, end_elas = k_final
    steps = sum(k_final) - sum(k_init)
    step_lb = (end_lb - start_lb) / steps
    step_elas = (end_elas - start_elas) / steps

    for k in tqdm(range(steps+1)):
        k_lb = int(start_lb + k * step_lb)
        k_elas = int(start_elas + k * step_elas)

        Cxy = pointmap2fmap(p2p, evecs_x[:, :k_lb], evecs_y[:, :k_lb])
        elas_Cxy = elas_pointmap2fmap(p2p, elas_evecs_x[:, :k_elas], elas_evecs_y[:, :k_elas], mass1, mass2)
        
        p2p = hybrid_fmap2pointmap(Cxy, evecs_x[:, :k_lb], evecs_y[:, :k_lb], elas_Cxy, elas_evecs_x[:, :k_elas], elas_evecs_y[:, :k_elas], mass1, mass2)
    
    return p2p

def trim_basis(data, n, elas_n):
    """
    trim the spectral operators (both LB ad Elastic) to specified numbers, intended for mixing up basis Fmap computation

    """
    # everthing has a batch dim
    evals = data['evals']
    elas_evals = data['elas_evals']
    data['evals'] = data['evals'][:, :n]
    data['evecs'] = data['evecs'][:, :, :n]
    data['evecs_trans'] = data['evecs_trans'][:, :n, :]
    if elas_n > 0:
        data['elas_evals'] = elas_evals[:, :elas_n]
        data['elas_evecs'] = data['elas_evecs'][:, :, :elas_n]

        # recompute the elastic evecs_trans
        mass = data['elas_mass'].squeeze(0)
        sqrtmass = torch.sqrt(mass)
        evecs = data['elas_evecs'].squeeze(0)
        def const_proj(evec, sqrtmass):
            # orthogonal projector for elastic basis
            sqrtM = torch.diag(sqrtmass)
            return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
        evecs_trans = const_proj(evecs, sqrtmass)

        data['elas_evecs_trans'] = evecs_trans.unsqueeze(0)

        # recompute the elastic Mk
        Mk = evecs.T @ torch.diag(mass) @ evecs
        data['elas_Mk'] = Mk.unsqueeze(0)

        # recompute sqrtMk and invsqrtMk
        sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness from LB
        sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
        invsqrtMk = torch.linalg.pinv(sqrtMk)
        data['elas_sqrtMk'] = sqrtMk.unsqueeze(0)
        data['elas_invsqrtMk'] = invsqrtMk.unsqueeze(0)

    return data


def corr2fmap(corr_x, corr_y, evecs_x, evecs_y):
    """
    Compute functional map from correspondences
    Cxy : shape x -> shape y
    """
    evecs_x_a = evecs_x[corr_x]
    evecs_y_a = evecs_y[corr_y]

    Cxy = torch.linalg.lstsq(evecs_y_a, evecs_x_a).solution
    return Cxy

def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()

def regularized_fmap_solve(A, B, evals_x, evals_y, lambda_=1e-3, resolvant_gamma=0.5):
    """Batched Inputs
    Compute the functional map matrix C by solving:
    min_C ||CA - B||^2 + lambda * ||D * C||^2
    where D is a mask based on eigenvalues that provides regularization.
    This is the same row-by-row solver implementation from DPFM and GeomFmaps papers.
    Bathed Output
    """
    D = get_mask(evals_x.flatten(), evals_y.flatten(), resolvant_gamma, A.device).unsqueeze(0)  # [B, K, K]

    A_t = A.transpose(1, 2)  # [B, C, K]
    A_A_t = torch.bmm(A, A_t)  # [B, K, K]
    B_A_t = torch.bmm(B, A_t)  # [B, K, K]
    
    C_i = []
    for i in range(evals_x.shape[1]):
        D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
        C = torch.bmm(torch.inverse(A_A_t + lambda_ * D_i), B_A_t[:, [i], :].transpose(1, 2))
        C_i.append(C.transpose(1, 2))
    C = torch.cat(C_i, dim=1)
    return C

def get_gt_fmap_with_regularization(corr_x, corr_y, evecs_x, evecs_y, evals_x, evals_y, lambda_=1e-3, resolvant_gamma=0.5, device="cuda"):
    """
    Un-Batched Inputs
    The functional map C is computed by solving:
    1. C = evecs_trans_y @ P @ evecs_x with additional regularization
    2. Minimizing: || evecs_y @ C = P @ evecs_x ||
    3. Equivalent to: || C.T @ evecs_y_a.T = evecs_x_a.T ||
    4. Final form: min_C || C.T @ coeff_Y = coeff_X ||
    Where:
    - P is the correspondence matrix
    - evecs_x, evecs_y are eigenvectors
    - coeff_Y, coeff_X are spectral coefficients
    Batched Output
    """

    # Un-Batched
    assert len(corr_x.shape) == 1 and len(corr_y.shape) == 1
    assert len(evecs_x.shape) == 2 and len(evecs_y.shape) == 2
    assert len(evals_x.shape) == 1 and len(evals_y.shape) == 1

    Nx, Ny = evecs_x.shape[0], evecs_y.shape[0]
    P = torch.zeros((Ny, Nx)).to(device)
    P[corr_y, corr_x] = 1

    evecs_x_a = P @ evecs_x
    evecs_y_a = evecs_y 
    coeff_X = evecs_x_a.t()
    coeff_Y = evecs_y_a.t()

    # Batched
    evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)
    coeff_X, coeff_Y = coeff_X.unsqueeze(0), coeff_Y.unsqueeze(0)
    C_transpose = regularized_fmap_solve(coeff_Y, coeff_X, evals_y, evals_x, lambda_, resolvant_gamma)

    C = C_transpose.transpose(1, 2)

    return C