import torch
import torch.nn as nn

from .csrc import k_nearest_neighbor, furthest_point_sampling
from .utils import Conv1dNormRelu, batch_indexing_channel_last, batch_indexing_channel_first


LEAKY_RATE = 0.1
use_bn = False


class SceneFlowEstimatorRefine(nn.Module):

    def __init__(self, feat_ch, hid_ch, cost_ch, flow_ch=3, channels=128, mlp=[128, 64], neighbors=12, clamp=[-200, 200], use_leaky=True):
        super(SceneFlowEstimatorRefine, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky

        self.flowModule = FlowRefineModule(neighbors, feat_ch, hid_ch, cost_ch, flow_ch, channels)
        last_channel = channels

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1dNormRelu(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, hid_feats, cost_volume, flow):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''

        new_points = self.flowModule(xyz, feats, hid_feats, cost_volume, flow)
        new_points = torch.tanh(new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)
        flow = self.fc(new_points)

        return new_points, flow.clamp(self.clamp[0], self.clamp[1])



class TransformerGroup(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim_in
        self.scale = qk_scale or dim_in ** -0.5

        self.q = nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=qkv_bias)

        self.pos_encoder = nn.Conv2d(4, dim_in, kernel_size=1, stride=1)

    def forward(self, xyz, neighbor_xyz, point, neighbor_points, neighbor_cost):
        '''
        xyz: [b, 3, n]
        neighbor_xyz: [b, 3, n, s]
        points: [b, c, n]
        neighbor_points: [b, c, n, s]
        '''

        B, C, N, S = neighbor_points.shape
        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)  # [b, 3, s, n]
        tmp = xyz_duplicate - neighbor_xyz
        xyz_norm = torch.norm(tmp, dim=1, keepdim=True)
        pos = torch.cat([tmp, xyz_norm], dim=1)

        q = point
        v = neighbor_cost

        pos_enc = self.pos_encoder(pos)
        k = self.k(neighbor_cost + pos_enc)
        v = self.v(v)

        q = q * self.scale
        attn = torch.einsum('b c n, b c n s->b n s', q, k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('b n s, b c n s->b c n', attn, v)

        return out


class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim_out
        self.scale = qk_scale or dim_out ** -0.5

        self.k = nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim_in + 3, dim_out, kernel_size=1, stride=1, bias=qkv_bias)

        self.pos_encoder = nn.Conv2d(4, dim_in, kernel_size=1, stride=1)

    def forward(self, xyz, neighbor_xyz, points, neighbor_points):
        '''
        xyz: [b, 3, n]
        neighbor_xyz: [b, 3, n, s]
        points: [b, c, n]
        neighbor_points: [b, c, n, s]
        '''

        B, C, N, S = neighbor_points.shape
        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)  # [b, 3, s, n]
        tmp = xyz_duplicate - neighbor_xyz
        xyz_norm = torch.norm(tmp, dim=1, keepdim=True)
        pos = torch.cat([tmp, xyz_norm], dim=1)

        q = points
        v = torch.cat([neighbor_points, tmp], dim=1)

        pos_enc = self.pos_encoder(pos)
        k = self.k(neighbor_points + pos_enc)
        v = self.v(v)

        q = q * self.scale
        attn = torch.einsum('b c n, b c n s->b n s', q, k)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('b n s, b c n s->b c n', attn, v)

        return out


class AttentionScore(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.scale = qk_scale or dim ** -0.5

        self.k = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)

        self.pos_encoder = nn.Conv2d(4, dim, kernel_size=1, stride=1)

    def forward(self, xyz, neighbor_xyz, points, neighbor_points):
        '''
        xyz: [b, 3, n]
        neighbor_xyz: [b, 3, n, s]
        points: [b, c, n]
        neighbor_points: [b, c, n, s]
        '''

        B, C, N, S = neighbor_points.shape
        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)  # [b, 3, s, n]
        tmp = xyz_duplicate - neighbor_xyz
        xyz_norm = torch.norm(tmp, dim=1, keepdim=True)
        pos = torch.cat([tmp, xyz_norm], dim=1)

        q = points
        pos_enc = self.pos_encoder(pos)
        k = self.k(neighbor_points + pos_enc)

        q = q * self.scale
        attn = torch.einsum('b c n, b c n s->b n s', q, k)
        attn = torch.softmax(attn, dim=-1)

        return attn


class Modulator(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=qkv_bias),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1, stride=1, bias=qkv_bias),
        )

        self.pos_encoder = nn.Conv2d(4, dim, kernel_size=1, stride=1)
        self.feat_encoder = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, xyz, neighbor_xyz, neighbor_feat, neighbor_cost):
        '''
        xyz: [b, 3, n]
        neighbor_xyz: [b, 3, n, s]
        neighbor_points: [b, c, n, s]
        neighbor_cost: [b, c, n, s]
        '''

        B, C, N, S = neighbor_feat.shape

        xyz = xyz.view(B, 3, N, 1)
        xyz_duplicate = xyz.repeat(1, 1, 1, S)  # [b, 3, s, n]
        tmp = xyz_duplicate - neighbor_xyz
        xyz_norm = torch.norm(tmp, dim=1, keepdim=True)
        pos = torch.cat([tmp, xyz_norm], dim=1)
        pos_enc = self.pos_encoder(pos)

        feat_enc = self.feat_encoder(neighbor_feat)
        cost_enc = neighbor_cost

        attn = self.mlp(feat_enc - cost_enc + pos_enc).squeeze()

        return attn



class FlowRefineModule(nn.Module):
    def __init__(self, nsample, feat_dim, hid_feat_dim, cost_dim, flow_dim, out_channels):
        super().__init__()
        self.nsample = nsample
        self.feat_dim = feat_dim
        self.hid_feat_dim = hid_feat_dim
        self.cost_dim = cost_dim
        self.flow_dim = flow_dim

        self.feat_attn = AttentionScore(feat_dim)
        self.hid_attn = AttentionScore(hid_feat_dim)
        self.cost_attn = AttentionScore(cost_dim)
        self.modulator = Modulator(feat_dim)

        self.proj = nn.Linear(2 * nsample, nsample)
        self.v = nn.Conv2d(feat_dim + hid_feat_dim + cost_dim + flow_dim, out_channels, 1, 1)


    def feature_split(self, points):
        b, c, n, s = points.shape
        start_ch = 0
        nei_feat = points[:, start_ch: start_ch + self.feat_dim]
        start_ch += self.feat_dim
        nei_hid_feat = points[:, start_ch: start_ch + self.hid_feat_dim]
        start_ch += self.hid_feat_dim
        nei_cost = points[:, start_ch: start_ch + self.cost_dim]
        start_ch += self.cost_dim
        nei_flow = points[:, start_ch: start_ch + self.flow_dim]

        return nei_feat, nei_hid_feat, nei_cost, nei_flow

    def forward(self, xyz, feat, hid_feat, cost, flow):
        '''
        b, c, n, k
        '''

        knn_idx = k_nearest_neighbor(xyz.permute(0, 2, 1), xyz.permute(0, 2, 1), self.nsample+1)[:, :, 1:]
        points = torch.cat([feat, hid_feat, cost, flow], dim=1)
        neighbor_xyz = batch_indexing_channel_first(xyz, knn_idx)
        neighbor_points = batch_indexing_channel_first(points, knn_idx)
        nei_feat, nei_hid_feat, nei_cost, nei_flow = self.feature_split(neighbor_points)

        # attn for different kinds of features
        attn_feat = self.feat_attn(xyz, neighbor_xyz, feat, nei_feat)
        attn_hid_feat = self.hid_attn(xyz, neighbor_xyz, hid_feat, nei_hid_feat)
        attn_cost = self.cost_attn(xyz, neighbor_xyz, cost, nei_cost)
        # attn_flow = self.flow_attn(xyz, neighbor_xyz, flow, nei_flow)
        ratio = self.modulator(xyz, neighbor_xyz, nei_feat, nei_cost)
        ratio = torch.sigmoid(ratio)

        # ratio = torch.sigmoid(attn_flow)
        attn_tmp = (1 - ratio) * attn_feat + ratio * attn_cost
        attn_total = torch.cat([attn_tmp, attn_hid_feat], dim=-1)
        attn_final = self.proj(attn_total)

        v = self.v(neighbor_points)
        new_points = torch.einsum('b n s, b c n s->b c n', attn_final, v)

        return new_points


