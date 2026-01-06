"""
This implementation follows the original DiffusionNet (Sharp et al.) and DPFM (Attaki et al.) implementations.
"""
import torch
import torch.nn as nn
from utils.registry import NETWORK_REGISTRY
from utils.geometry_util import compute_wks_autoscale, compute_hks_autoscale, data_augmentation, hash_arrays, torch2np
from utils.temp_seed_util import temp_seed
def to_basis(values, basis, massvec):
    """
    Transform data in to an orthonormal basis (where orthonormal is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values, basis):
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        return utils.cmatmul(utils.ensure_complex(basis), utils.ensure_complex(values)) 
    else:
        return torch.matmul(basis, values)


class LearnedTimeDiffusion(nn.Module):
    """
    Applied diffusion with learned time per-channel.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i * t) * f_in
    """
    def __init__(self, in_channels, method='spectral'):
        """
        Args:
            in_channels (int): number of input channels.
            method (str, optional): method to perform time diffusion. Default 'spectral'.
        """
        super(LearnedTimeDiffusion, self).__init__()
        assert method in ['spectral', 'implicit_dense'], f'Invalid method: {method}'
        self.in_channels = in_channels
        self.diffusion_time = nn.Parameter(torch.Tensor(in_channels))
        self.method = method
        # init as zero
        nn.init.constant_(self.diffusion_time, 0.0)

    def forward(self, feat, L, mass, evals, evecs):
        """
        Args:
            feat (torch.Tensor): feature vector [B, V, C].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            mass (torch.Tensor): diagonal elements in mass matrix [B, V].
            evals (torch.Tensor): eigenvalues of Laplacian matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian matrix [B, V, K].
        Returns:
            feat_diffuse (torch.Tensor): diffused feature vector [B, V, C].
        """
        # project times to the positive half-space
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(self.diffusion_time, min=1e-8)

        assert feat.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat.shape[-1]}'

        if self.method == 'spectral':
            # Transform to spectral
            feat_spec = to_basis(feat, evecs, mass)

            # Diffuse
            diffuse_coefs = torch.exp(-evals.unsqueeze(-1) * self.diffusion_time.unsqueeze(0))
            feat_diffuse_spec = diffuse_coefs * feat_spec

            # Transform back to feature
            feat_diffuse = from_basis(feat_diffuse_spec, evecs)

        else: # 'implicit_dense'
            V = feat.shape[-2]

            # Form the dense matrix (M + tL) with dims (B, C, V, V)
            mat_dense = L.to_dense().unsuqeeze(1).expand(-1, self.in_channels, -1, -1).clone()
            mat_dense *= self.diffusion_time.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            mat_dense += torch.diag_embed(mass).unsqueeze(1)

            # Factor the system
            cholesky_factors = torch.linalg.cholesky(mat_dense)

            # Solve the system
            rhs = feat * mass.unsqueeze(-1)
            rhsT = rhs.transpose(1, 2).unsqueeze(-1)
            sols = torch.cholesky_solve(rhsT, cholesky_factors)
            feat_diffuse = sols.squeeze(-1).transpose(1, 2)

        return feat_diffuse


class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors.
    Uses a learned complex-linear layer to keep dimension down.
    """
    def __init__(self, in_channels, with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            with_gradient_rotations (bool, optional): whether with gradient rotations. Default True.
        """
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """
        Args:
            feat_in (torch.Tensor): input feature vector (B, V, C, 2).
        Returns:
            feat_out (torch.Tensor): output feature vector (B, V, C)
        """
        feat_a = feat_in

        if self.with_gradient_rotations:
            feat_real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            feat_img_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            feat_real_b = self.A(feat_in[..., 0])
            feat_img_b = self.A(feat_in[..., 1])

        feat_out = feat_a[..., 0] * feat_real_b + feat_a[..., 1] * feat_img_b

        return torch.tanh(feat_out)


class MiniMLP(nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes
    """
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU, name='miniMLP'):
        """
        Args:
            layer_sizes (List): list of layer size.
            dropout (bool, optional): whether use dropout. Default False.
            activation (nn.Module, optional): activation function. Default ReLU.
            name (str, optional): module name. Default 'miniMLP'
        """
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            # Dropout Layer
            if dropout and i > 0:
                self.add_module(
                    name + '_dropout_{:03d}'.format(i),
                    nn.Dropout(p=0.5)
                )

            # Affine Layer
            self.add_module(
                name + '_linear_{:03d}'.format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )

            # Activation Layer
            if not is_last:
                self.add_module(
                    name + '_activation_{:03d}'.format(i),
                    activation()
                )


class DiffusionNetBlock(nn.Module):
    """
    Building Block of DiffusionNet.
    """
    def __init__(self, in_channels, mlp_hidden_channels,
                 dropout=True,
                 diffusion_method='spectral',
                 with_gradient_features=True,
                 with_gradient_rotations=True):
        """
        Args:
            in_channels (int): number of input channels.
            mlp_hidden_channels (List): list of mlp hidden channels.
            dropout (bool, optional): whether use dropout in MLP. Default True.
            with_gradient_features (bool, optional): whether use spatial gradient feature. Default True.
            with_gradient_rotations (bool, optional): whether use spatial gradient rotation. Default True.
        """
        super(DiffusionNetBlock, self).__init__()

        self.in_channels = in_channels
        self.mlp_hidden_channels = mlp_hidden_channels
        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.in_channels, method=diffusion_method)

        # concat of both diffused features and original features
        self.mlp_in_channels = 2 * self.in_channels

        # Spatial gradient block
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.in_channels,
                                                             with_gradient_rotations=self.with_gradient_rotations)
            # concat of gradient features
            self.mlp_in_channels += self.in_channels

        # MLP block
        self.mlp = MiniMLP([self.mlp_in_channels] + self.mlp_hidden_channels + [self.in_channels], dropout=self.dropout)

    def forward(self, feat_in, mass, L, evals, evecs, gradX, gradY):
        """
        Args:
            feat_in (torch.Tensor): input feature vector [B, V, C].
            mass (torch.Tensor): diagonal elements of mass matrix [B, V].
            L (torch.SparseTensor): sparse Laplacian matrix [B, V, V].
            evals (torch.Tensor): eigenvalues of Laplacian Matrix [B, K].
            evecs (torch.Tensor): eigenvectors of Laplacian Matrix [B, V, K].
            gradX (torch.SparseTensor): real part of gradient matrix [B, V, V].
            gradY (torch.SparseTensor): imaginary part of gradient matrix [B, V, V].
        """

        B = feat_in.shape[0]
        assert feat_in.shape[-1] == self.in_channels, f'Expected feature channel: {self.in_channels}, but got: {feat_in.shape[-1]}'

        # Diffusion block
        feat_diffuse = self.diffusion(feat_in, L, mass, evals, evecs)

        # Compute gradient features
        if self.with_gradient_features:
            # Compute gradient
            feat_grads = []
            for b in range(B):
                # gradient after diffusion
                feat_gradX = torch.mm(gradX[b, ...], feat_diffuse[b, ...])
                feat_gradY = torch.mm(gradY[b, ...], feat_diffuse[b, ...])

                feat_grads.append(torch.stack((feat_gradX, feat_gradY), dim=-1))
            feat_grad = torch.stack(feat_grads, dim=0) # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse, feat_grad_features), dim=-1)
        else:
            # Stack inputs to MLP
            feat_combined = torch.cat((feat_in, feat_diffuse), dim=-1)

        # MLP block
        feat_out = self.mlp(feat_combined)

        # Skip connection
        feat_out = feat_out + feat_in

        return feat_out

@NETWORK_REGISTRY.register()
class DiffusionNet_B(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        n_block=4,
        last_activation=None,
        outputs_at="vertices",
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        diffusion_method="spectral",
        input_type="xyz",
        augmentation={'train': {}, 'test': {}}
    ):
        """
        Construct a DiffusionNet.
        Parameters:
            in_channels (int):                     input dimension
            out_channels (int):                    output dimension
            last_activation (func)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces'].
            (default 'vertices', aka points for a point cloud)
            hidden_channels (int):                  dimension of internal DiffusionNet blocks (default: 128)
            n_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [hidden_channels, hidden_channels])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (string):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0,
            saving precompute.
            with_gradient_features (bool):  if True, use gradient features (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a rotation of each gradient.
            Set to True if your surface has consistently oriented normals, and False otherwise (default: True)
        """

        super(DiffusionNet_B, self).__init__()

        # Store parameters
        self.input_type = input_type
        # augmentation
        self.DEFAULT_TRAIN_AUGMENTATIONS = {
            'rot_x': 30.0,
            'rot_y': 30.0,
            'rot_z': 30.0,
            'std': 0.01,
            'noise_clip': 0.05,
            'scale_min': 0.9,
            'scale_max': 1.1
        }
        self.DEFAULT_TEST_AUGMENTATIONS = {
            'rot_x': 0.0,
            'rot_y': 0.0,
            'rot_z': 0.0,
            'std': 0.0,
            'noise_clip': 0.0,
            'scale_min': 1.0,
            'scale_max': 1.0
        }
        self.train_augmentation = {**self.DEFAULT_TRAIN_AUGMENTATIONS, **(augmentation.get("train", {}) or {})}
        self.test_augmentation = {**self.DEFAULT_TEST_AUGMENTATIONS, **(augmentation.get("test", {}) or {})}

        if self.input_type == 'xyz':
            print("Settings:")
            print(f"  Input type: {self.input_type}")
            print(f"  Train augmentations: {self.train_augmentation}")
            print(f"  Test  augmentations: {self.test_augmentation}")

        # Basic parameters
        self.C_in = in_channels
        self.C_out = out_channels
        self.C_width = hidden_channels
        self.N_block = n_block

        # Outputs
        self.last_activation = last_activation
        self.outputs_at = outputs_at
        if outputs_at not in ["vertices", "edges", "faces"]:
            raise ValueError("invalid setting for outputs_at")

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [self.C_width, self.C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Diffusion
        self.diffusion_method = diffusion_method
        if diffusion_method not in ["spectral", "implicit_dense"]:
            raise ValueError("invalid setting for diffusion_method")

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_linear = nn.Linear(self.C_in, self.C_width)
        self.last_linear = nn.Linear(self.C_width, self.C_out)

        # DiffusionNet blocks
        blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlock(
                in_channels=hidden_channels,
                mlp_hidden_channels=self.mlp_hidden_dims,
                dropout=dropout,
                diffusion_method=diffusion_method,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations
            )
            blocks += [block]
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        data,
        x_in=None
    ):
        """
        A forward pass on the DiffusionNet.
        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].
        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet,
        not all are strictly necessary.
        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        mass, L, evals, evecs, gradX, gradY = data['operators']['mass'], data['operators']['L'], \
                                                    data['operators']['evals'], data['operators']['evecs'], \
                                                    data['operators']['gradX'], data['operators']['gradY']
        if x_in is None:
            if self.input_type == 'xyz':
                x_in = data['xyz']
                if self.training:
                    x_in = data_augmentation(x_in, **self.train_augmentation)
                else: # ensure reproducible rotations for control experiments
                    hash = hash_arrays([torch2np(data['verts']), torch2np(data['faces'])]) 
                    seed = int(hash, 16) % (2**32)
                    with temp_seed(seed):
                        x_in = data_augmentation(x_in, **self.test_augmentation)
            elif self.input_type == 'wks':
                x_in = compute_wks_autoscale(evals, evecs, mass)
            elif self.input_type == 'hks':
                x_in = compute_hks_autoscale(evals, evecs)
            elif self.input_type == 'dino':
                x_in = data['dino']
            else:
                x_in = data[self.input_type]
        if x_in.ndim != 3:
            raise ValueError(f"Input tensor x_in must be batched with shape [B,N,C], but got shape {list(x_in.shape)}")
        

        # Apply the first linear layer
        x = self.first_linear(x_in)

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x = self.last_linear(x)

        # Remap output to faces/edges if requested
        if self.outputs_at == "vertices":
            x_out = x

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        return x_out
