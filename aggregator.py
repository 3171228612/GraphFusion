import torch
import torch.nn as nn
import torch.nn.functional as F

from psgformer.ops import ballquery_batchflat, furthestsampling_batchflat
from psgformer.pointnet2.pointnet2_utils import ball_query, furthest_point_sample
from .module_utils import Conv1d, SharedMLP

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class LocalAggregator(nn.Module):
    def __init__(
        self,
        mlp_dim: int = 32,
        n_sample: int = 1024,
        radius: float = 0.4,
        n_neighbor: int = 128,
        n_neighbor_post: int = 128,
        bn: bool = True,
        use_xyz: bool = True,
    ) -> None:
        super().__init__()

        self.n_sample = n_sample
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.n_neighbor_post = n_neighbor_post
        self.use_xyz = use_xyz

        self.radius_post = 2 * radius

        mlp_spec1 = [mlp_dim, mlp_dim, mlp_dim * 2]
        mlp_spec1[0] += 3

        self.mlp_module1 = SharedMLP(mlp_spec1, bn=bn)

        mlp_spec2 = [mlp_dim * 2, mlp_dim * 2]
        if use_xyz and len(mlp_spec2) > 0:
            mlp_spec2[0] += 3
        self.mlp_module2 = SharedMLP(mlp_spec2, bn=bn, activation=None)

        mlp_module3 = [
            Conv1d(in_size=mlp_dim * 2, out_size=mlp_dim * 2 * 4, bn=True),
            Conv1d(
                in_size=mlp_dim * 2 * 4,
                out_size=mlp_dim * 2,
                bn=True,
                activation=None,
            ),
        ]

        self.mlp_module3 = nn.Sequential(*mlp_module3)

        self.skip_act = nn.ReLU()

    def forward(self, locs, feats, batch_offsets=None, batch_size=1, sampled_before=False):

        if len(locs.shape) == 2:
            return self.forward_batchflat(locs, feats, batch_offsets, batch_size, sampled_before=sampled_before)
        else:
            raise RuntimeError("Invalid shape to LocalAggregator")

    def forward_batchflat(self, locs, feats, batch_offsets, batch_size, sampled_before=False):
        fps_offsets = torch.arange(
            0, self.n_sample * (batch_size + 1), self.n_sample, dtype=torch.int, device=locs.device
        )
        fps_inds = furthestsampling_batchflat(locs, batch_offsets, fps_offsets)

        if fps_inds.max().item() >= locs.size(0):
            print(f"Warning: Invalid index found in fps_inds at epoch , batch {batch_size}. Attempting to fix.")
            fps_inds = fps_inds.clamp(max=locs.size(0) - 1)

        if fps_inds.size(0) < batch_size * self.n_sample:
            print(f"Warning: Not enough points in fps_locs_float at epoch , batch {batch_size}. Padding with zeros.")
            padding = torch.zeros(batch_size * self.n_sample - fps_inds.size(0), dtype=torch.int, device=locs.device)
            fps_inds = torch.cat([fps_inds, padding])

        fps_locs_float = locs[fps_inds.long(), :]  # m, 3

        neighbor_inds = ballquery_batchflat(
            self.radius, self.n_neighbor, locs, fps_locs_float, batch_offsets, fps_offsets
        )  # m, nsample
        neighbor_inds = neighbor_inds.reshape(-1).long()

        grouped_xyz = torch.gather(locs, 0, neighbor_inds[:, None].expand(-1, locs.shape[-1])).reshape(
            batch_size * self.n_sample, self.n_neighbor, locs.shape[-1]
        )  # m, nsample, 3
        grouped_xyz = (grouped_xyz - fps_locs_float[:, None, :]) / self.radius

        grouped_features = torch.gather(feats, 0, neighbor_inds[:, None].expand(-1, feats.shape[-1])).reshape(
            batch_size * self.n_sample, self.n_neighbor, feats.shape[-1]
        )  # m, nsample, 3

        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)  # m, nsample, C

        grouped_features = grouped_features.reshape(batch_size, self.n_sample, self.n_neighbor, -1)

        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # B, C, nqueries, npoints

        # Applying MLP and max_pool2d
        new_features = self.mlp_module1(grouped_features)  # (B, mlp[-1], npoint, nsample)

        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)


        identity = new_features

        # Second neighbor query
        fps_locs_float = fps_locs_float.reshape(batch_size, self.n_sample, -1).contiguous()
        fps_inds = fps_inds.reshape(batch_size, self.n_sample)

        neighbor_inds2 = ball_query(self.radius_post, self.n_neighbor_post, fps_locs_float, fps_locs_float)  # B, m, m1
        neighbor_inds2 = neighbor_inds2.reshape(batch_size, -1).long()  # b, m*m1

        grouped_xyz2 = torch.gather(
            fps_locs_float, 1, neighbor_inds2[:, :, None].expand(-1, -1, fps_locs_float.shape[-1])
        ).reshape(
            batch_size, self.n_sample, self.n_neighbor_post, fps_locs_float.shape[-1]
        )  # B, m, m1, 3
        grouped_xyz2 = (grouped_xyz2 - fps_locs_float[:, :, None, :]) / self.radius_post
        grouped_xyz2 = grouped_xyz2.permute(0, 3, 1, 2)

        #new_feats_reshaped = new_features.reshape(batch_size, self.n_sample, -1)
        grouped_features2 = torch.gather(
            new_features, 2, neighbor_inds2[:, None, :].expand(-1, new_features.shape[1], -1)
        ).reshape(
            batch_size, new_features.shape[1], self.n_sample, self.n_neighbor_post
        )  # m, nsample, 3

        grouped_features2 = torch.cat([grouped_xyz2, grouped_features2], dim=1)  
        
        # Applying MLP and max_pool2d
        new_features2 = self.mlp_module2(grouped_features2)  
        
        new_features2 = F.max_pool2d(new_features2, kernel_size=[1, new_features2.size(3)])  # (B, mlp[-1], npoint, 1)
        new_features2 = new_features2.squeeze(-1)  # (B, mlp[-1], npoint)

        new_features3 = self.mlp_module3(new_features2)
        
        feats = self.skip_act(new_features3 + identity)
        feats = feats.permute(0,2,1) #B, N, C
        # Reshaping the output to [B*N, C]
        new_features4 = feats.reshape(-1, feats.size(-1))  # (B*N, mlp2[-1])
       
        
        return new_features4, fps_inds, fps_locs_float




