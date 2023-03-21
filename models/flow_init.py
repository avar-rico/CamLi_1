import torch
import torch.nn as nn

from models.csrc import k_nearest_neighbor
from models.pointconv import PointConvNoSampling
from models.utils import Conv1dNormRelu, batch_indexing_channel_first
from models.update import SelfAttention


class SceneFlowEstimatorInit(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels=128, mlp = [128, 64], neighbors = 32, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorInit, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky

        assert feat_ch == feat_ch
        self.flowModule = FlowInitModule(neighbors, feat_ch, channels)
        last_channel = channels

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1dNormRelu(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''

        new_points = self.flowModule(xyz, feats, cost_volume)
        new_points = torch.tanh(new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)
        flow = self.fc(new_points)

        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class FlowInitModule(nn.Module):
    def __init__(self, nsample, dim_in, dim_out):
        super().__init__()
        self.nsample = nsample

        self.mod_add = PointConvNoSampling(dim_in, dim_in, norm='batch_norm', k=16)
        self.attn1 = SelfAttention(dim_in, dim_out)

    def forward(self, xyz, points, cost):

        knn_idx = k_nearest_neighbor(xyz.permute(0, 2, 1), xyz.permute(0, 2, 1), self.nsample)
        neighbor_xyz = batch_indexing_channel_first(xyz, knn_idx)

        diff = cost - points
        cost_add = self.mod_add(xyz, diff)
        cost_mod = cost + cost_add
        grouped_cost = batch_indexing_channel_first(cost_mod, knn_idx)
        new_points = self.attn1(xyz, neighbor_xyz, cost_mod, grouped_cost)

        return new_points