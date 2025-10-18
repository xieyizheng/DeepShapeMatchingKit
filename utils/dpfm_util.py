# util functions from DPFM repository
import os
import torch
from utils.geometry_util import get_all_operators, torch2np, hash_arrays
import os.path as osp
import numpy as np
def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
def nn_interpolate(desc, xyz, dists, idx, idf):
    # bug could be that idx is already a idf based index

    # that is indeed the bug, and further more, if we are to change sampling ratio after shape dataset cache, we need to regenerte idx

    # but anyway, this is fixed now, so all working as expected and we should expect a good performance now
    xyz = xyz.unsqueeze(0)
    B, N, _ = xyz.shape
    
    mask = idx >= idf.size(0)

    dists[mask] = 1e8 # sets the distances of the elements in idx that are not in idf to a large number
    
    mask = torch.argsort(dists, dim=-1, descending=False)[:, :, :3] # Sorts mask to bring indices where idx matches idf to the front. It then slices the first three indices for each row in each batch
    
    dists, idx = torch.gather(dists, 2, mask), torch.gather(idx, 2, mask) # filters dists and idx to only include distances and indices for the top 3 matches
    
    dists, idx = dists.to(desc.device), idx.to(desc.device)

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated_points = torch.sum(index_points(desc, idx) * weight.view(B, N, 3, 1), dim=2)

    return interpolated_points
def cache_sample_idx(data, cross_sampling_ratio=1.0, cache_dir=None):
    if cross_sampling_ratio == 1.0: # dont' need cache because no interpolation needed
        return None
    data_x, data_y = data['first'], data['second']
    if 'sample_idx' not in data_x.keys():
        cache_dir = cache_dir or data_x.get('cache_dir', None)
        # perform cache operation
        verts, faces = data_x['verts'], data_x['faces'] # batched with 1
        verts, faces = verts.squeeze(0), faces.squeeze(0)
        dists, idx0, idx1 = get_sample_idx(verts, faces, ratio=cross_sampling_ratio, cache_dir=cache_dir)
        # batch with 1
        # dists, idx0, idx1 = dists.unsqueeze(0), idx0.unsqueeze(0), idx1.unsqueeze(0)
        data_x['sample_idx'] = (idx0, idx1, dists)
    if 'sample_idx' not in data_y.keys():
        cache_dir = cache_dir or data_y.get('cache_dir', None)
        # perform cache operation
        verts, faces = data_y['verts'], data_y['faces'] # batched with 1
        verts, faces = verts.squeeze(0), faces.squeeze(0)
        dists, idx0, idx1 = get_sample_idx(verts, faces, ratio=cross_sampling_ratio, cache_dir=cache_dir)
        # batch with 1
        # dists, idx0, idx1 = dists.unsqueeze(0), idx0.unsqueeze(0), idx1.unsqueeze(0)
        data_y['sample_idx'] = (idx0, idx1, dists)
def get_sample_idx(verts, faces, ratio=1.0, cache_dir=None):
    assert verts.dim() == 2, 'Please call get_all_operators() for a batch of vertices'
    device = verts.device
    dtype = verts.dtype
    int_dtype = torch.int32
    verts_np = torch2np(verts)
    faces_np = torch2np(faces) if faces is not None else None

    if np.isnan(verts_np).any():
        raise ValueError('detect NaN vertices.')

    found = False
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        hash_key_str = "sample_idx" + str(ratio) + str(hash_arrays((verts_np, faces_np))) # sample idx hash string

        # Search through buckets with matching hashes.
        # When the loop exits,
        # this is the bucket index of the file we should write to.
        i_cache = 0
        while True:
            # From the name of the file to check
            search_path = osp.join(cache_dir, hash_key_str+'_'+str(i_cache)+'.npz')

            try:
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile['verts']
                cache_faces = npzfile['faces']

                # If the cache doesn't match, keep searching
                if (not np.array_equal(verts, cache_verts)) or (not np.array_equal(faces, cache_faces)):
                    i_cache += 1
                    print('collision detected')
                    continue

                # this entry matches. return it.
                dists = npzfile['frames']
                idx0 = npzfile['mass']
                idx1 = npzfile['evals']

                dists = torch.from_numpy(dists).to(device=device, dtype=dtype)
                idx0 = torch.from_numpy(idx0).to(device=device, dtype=int_dtype)
                idx1 = torch.from_numpy(idx1).to(device=device, dtype=int_dtype)

                found = True
                # print(f"cache hit, loading sample idx for DPFM from cache with ratio : {ratio}")
                break
            except FileNotFoundError:
                # not found, create a new file
                break

    if not found:
        # recompute
        print(f"cache miss, recomputing sample idx for DPFM with ratio : {ratio}")
        idx0 = farthest_point_sample(verts.t(), ratio=ratio) # here the ratio is very important to keep the idx sanity!
        dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
        dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()

        dtype_np = np.float32
        int_dtype_np = np.int32

        # save
        if cache_dir:
            dists_np = torch2np(dists).astype(dtype_np)
            idx0_np = torch2np(idx0).astype(int_dtype_np)
            idx1_np = torch2np(idx1).astype(int_dtype_np)

            np.savez(
                search_path,
                verts=verts_np,
                faces=faces_np,
                frames=dists_np,
                mass=idx0_np,
                evals=idx1_np,
            )

    return dists, idx0, idx1

def farthest_point_sample(xyz, ratio):
    xyz = xyz.t().unsqueeze(0)
    npoint = int(ratio * xyz.shape[1])
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids[0]

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist