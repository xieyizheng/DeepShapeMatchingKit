try:
    import pyshot
except ImportError:
    print("Warning: Failed to import pyshot. Some functionality may be limited.")

from utils.geometry_util import hash_arrays, torch2np


import numpy as np
import torch


import os
import os.path as osp

from utils.cache_util import get_cached_compute

def compute_shot_desc(verts, faces, radius=0.3, local_rf_radius=0.3, min_neighbors=3, n_bins=5):
    """
    Compute SHOT descriptors for a mesh.
    
    Args:
        verts (np.ndarray): Vertex positions [V, 3]
        faces (np.ndarray): Face indices [F, 3]
        radius (float): Radius for descriptor computation
        local_rf_radius (float): Radius for local reference frame
        min_neighbors (int): Minimum number of neighbors
        n_bins (int): Number of bins in histogram
        
    Returns:
        np.ndarray: SHOT descriptors
    """
    return pyshot.get_descriptors(
        verts.astype(np.float64),
        faces,
        radius=radius,
        local_rf_radius=local_rf_radius,
        min_neighbors=min_neighbors,
        n_bins=n_bins,
        double_volumes_sectors=True,
        use_interpolation=True,
        use_normalization=True
    )

def get_shot_desc(verts, faces, cache_dir=None, overwrite_cache=False):
    """
    Get SHOT descriptors for a mesh, using caching if possible.
    
    Args:
        verts (torch.Tensor): Vertex positions [V, 3]
        faces (torch.Tensor): Face indices [F, 3] 
        cache_dir (str, optional): Directory to cache results. Default None.
        overwrite_cache (bool): Whether to overwrite existing cache. Default False.
        
    Returns:
        torch.Tensor: SHOT descriptors
    """
    return get_cached_compute(compute_shot_desc, verts, faces, cache_dir=cache_dir,
                            radius=0.3, local_rf_radius=0.3, min_neighbors=3, n_bins=5)