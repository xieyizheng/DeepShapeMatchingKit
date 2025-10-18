import os
import re
import torch
import scipy.linalg
from utils.tensor_util import to_numpy
from utils.geometry_util import get_operators
from utils.shape_util import get_geodesic_distmat
from utils.dino_util import get_shape_dino_features
from utils.shot_util import get_shot_desc
from utils.elastic_util import get_elas_operators

def get_shape_operators_and_data(item, cache_dir, config):
    """Get spectral and elastic operators for a shape."""
    verts, faces = item['verts'], item['faces']

    if config.get('return_evecs', True):
        item = get_spectral_ops(item, num_evecs=config.get('num_evecs', 200), cache_dir=os.path.join(cache_dir, 'diffusion'))
    
    if config.get('return_elas_evecs', False):
        bending_weight = config.get('bending_weight', 1e-2)
        item = get_elas_spectral_ops(item, num_evecs=config.get('num_evecs', 200), bending_weight=bending_weight, cache_dir=os.path.join(cache_dir, f'elastic_harmonic_{bending_weight}'))
        
    if config.get('return_dist', False):
        item['dist'] = get_geodesic_distmat(verts, faces, cache_dir=os.path.join(cache_dir, 'dist'))
    
    if config.get('return_dino', False):
        item['dino'] = get_shape_dino_features(verts, faces, cache_dir=os.path.join(cache_dir, 'dino')) * 0.5

    if config.get('return_shot', False):
        item['shot'] = get_shot_desc(verts, faces, cache_dir=os.path.join(cache_dir, 'shot'))
    
    item['xyz'] = verts
    if config.get('xyz_reorient', None):
        from utils.visualization_util import get_orientation_calibration_matrix
        orient_calib_R_src = get_orientation_calibration_matrix(config['xyz_reorient']['src_up'], config['xyz_reorient']['src_front'])
        orient_calib_R_dst = get_orientation_calibration_matrix(config['xyz_reorient']['dst_up'], config['xyz_reorient']['dst_front'])
        item['xyz'] = item['xyz'] @ orient_calib_R_src @ orient_calib_R_dst.T
    return item






def get_spectral_ops(item, num_evecs, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    
    # Use max of num_evecs and 128 for get_operators
    k = max(num_evecs, 128)
    _, mass, L, evals, evecs, gradX, gradY = get_operators(item['verts'], item.get('faces'),
                                                   k=k,
                                                   cache_dir=cache_dir)
    
    # Store 128-length operators in diffusion dict
    item['operators'] = {
        'evecs': evecs[:, :128],
        'evecs_trans': (evecs.T * mass[None])[:128],
        'evals': evals[:128],
        'mass': mass,
        'L': L,
        'gradX': gradX, 
        'gradY': gradY
    }
    evecs_trans = evecs.T * mass[None]
    item['evecs'] = evecs[:, :num_evecs]
    item['evecs_trans'] = evecs_trans[:num_evecs]
    item['evals'] = evals[:num_evecs]
    item['mass'] = mass
    item['L'] = L

    return item


def get_elas_spectral_ops(item, num_evecs, bending_weight=1e-2, cache_dir=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    mass, evals, evecs = get_elas_operators(item['verts'], item.get('faces'),
                                                   k=num_evecs, bending_weight=bending_weight, 
                                                   cache_dir=cache_dir)
    # evecs = treat_nan(evecs, item['verts'], item['name'])
    mass = mass + 1e-8 * mass.mean()
    sqrtmass = torch.sqrt(mass)
    
    def const_proj(evec, sqrtmass):
        # orthogonal projector for elastic basis
        sqrtM = torch.diag(sqrtmass)
        return torch.linalg.pinv(sqrtM @ evec) @ sqrtM
    evecs_trans = const_proj(evecs[:, :num_evecs], sqrtmass)
    Mk = evecs.T @ torch.diag(mass) @ evecs
    item['elas_evecs'] = evecs[:, :num_evecs]
    item['elas_evecs_trans'] = evecs_trans[:num_evecs]
    item['elas_evals'] = evals[:num_evecs]
    item['elas_mass'] = mass
    item['elas_Mk'] = Mk
    sqrtMk = scipy.linalg.sqrtm(to_numpy(Mk)).real #numerical weirdness 
    sqrtMk = torch.tensor(sqrtMk).float().to(Mk.device)
    invsqrtMk = torch.linalg.pinv(sqrtMk)
    item['elas_sqrtMk'] = sqrtMk
    item['elas_invsqrtMk'] = invsqrtMk
    return item














def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)



