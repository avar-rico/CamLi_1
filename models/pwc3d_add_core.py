import torch
import torch.nn as nn
from .pointconv import PointConvNoSampling, PointConvDownSampling
from .utils import MLP1d, MLP2d, batch_indexing_channel_first
from .csrc import k_nearest_neighbor, furthest_point_sampling

use_bn = False

class TransformerGroup(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim ** -0.5

        self.q = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)

        self.pos_encoder = nn.Conv2d(4, dim, kernel_size=1, stride=1)


    def forward(self, xyz, neighbor_xyz, points, neighbor_points):
        '''
        xyz: [b, 3, N]
        neighbor_xyz: [b, 3, n, k]
        points: [b, c, n]
        neighbor_points: [b, c, n, k]
        '''

        B, C, N, S = neighbor_points.shape
        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)

        tmp = xyz_duplicate - neighbor_xyz
        xyz_norm = torch.norm(tmp, dim=1, keepdim=True)
        pos = torch.cat([tmp, xyz_norm], dim=1)

        q = self.q(points)

        pos_enc = self.pos_encoder(pos)
        k = self.k(neighbor_points + pos_enc)
        v = self.v(neighbor_points)

        q = q * self.scale
        attn = torch.einsum('b c n, b c n s->b n s', q, k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('b n s, b c n s->b c n', attn, v)

        return out


class TransformerExtendGroup(nn.Module):
    def __init__(self, dim, out_dim, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim ** -0.5

        self.q = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.k = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.v = nn.Conv2d(2*dim+3, out_dim, kernel_size=1, stride=1, bias=qkv_bias)

        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 4, kernel_size=1, stride=1, bias=qkv_bias),
            nn.ReLU(),
            nn.Conv2d(out_dim // 4, 1, kernel_size=1, stride=1, bias=qkv_bias),
        )

        self.pos_encoder = nn.Conv2d(5, out_dim, kernel_size=1, stride=1)


    def forward(self, xyz, neighbor_xyz, points, neighbor_points):
        '''
        xyz: [b, 3, N]
        neighbor_xyz: [b, 3, n, k]
        points: [b, c, n]
        neighbor_points: [b, c, n, k]
        '''

        B, C, N, S = neighbor_points.shape
        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)

        points_duplicate = points.view(B, C, N, 1).repeat(1, 1, 1, S)
        feat_norm = torch.norm(points_duplicate - neighbor_points, dim=1, keepdim=True)
        xyz_norm = torch.norm(xyz_duplicate - neighbor_xyz, dim=1, keepdim=True)
        pos = torch.cat([feat_norm, xyz_norm, xyz_duplicate - neighbor_xyz], dim=1)
        v = torch.cat([points_duplicate, neighbor_points, xyz_duplicate - neighbor_xyz], dim=1)
        pos_enc = self.pos_encoder(pos)

        q = self.q(points_duplicate)
        v = self.v(v)
        k = self.k(v + pos_enc)

        attn = self.mlp(q-k).squeeze()
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('b n s, b c n s->b c n', attn, v)

        return out


class Correlation3D(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()

        self.k = k

        self.group1 = TransformerExtendGroup(in_channels, out_channels)
        self.group2 = TransformerGroup(out_channels)

        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, xyz1, feat1, xyz2, feat2):
        """
        :param xyz1: [batch_size, 3, n_points]
        :param feat1: [batch_size, in_channels, n_points]
        :param xyz2: [batch_size, 3, n_points]
        :param feat2: [batch_size, in_channels, n_points]
        :param knn_indices_1in1: for each point in xyz1, find its neighbors in xyz1, [batch_size, n_points, k]
        :return cost volume for each point in xyz1: [batch_size, n_cost_channels, n_points]
        """
        batch_size, in_channels, n_points = feat1.shape

        # Step1: for each point in xyz1, find its neighbors in xyz2
        knn_indices_1in2 = k_nearest_neighbor(input_xyz=xyz2, query_xyz=xyz1, k=self.k)
        # knn_xyz2: [bs, 3, n_points, k]
        knn_xyz2 = batch_indexing_channel_first(xyz2, knn_indices_1in2)
        # knn_features2: [bs, in_channels, n_points, k]
        knn_features2 = batch_indexing_channel_first(feat2, knn_indices_1in2)
        # p2n_cost (point-to-neighbor cost): [bs, out_channels, n_points]
        p2n_cost = self.group1(xyz1, knn_xyz2, feat1, knn_features2)
        p2n_cost = self.relu(self.bn(p2n_cost))

        # Step2: for each point in xyz1, find its neighbors in xyz1
        knn_indices_1in1 = k_nearest_neighbor(input_xyz=xyz1, query_xyz=xyz1, k=self.k)  # [bs, n_points, k]
        # knn_xyz1: [bs, 3, n_points, k]
        knn_xyz1 = batch_indexing_channel_first(xyz1, knn_indices_1in1)
        # n2n_cost: [bs, out_channels, n_points, k]
        n2n_cost = batch_indexing_channel_first(p2n_cost, knn_indices_1in1)
        # n2n_cost (neighbor-to-neighbor cost): [bs, out_channels, n_points]
        n2n_cost = self.group2(xyz1, knn_xyz1, p2n_cost, n2n_cost)
        n2n_cost = self.relu(n2n_cost)

        return n2n_cost