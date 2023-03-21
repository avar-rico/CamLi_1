import torch
import torch.nn as nn
from .utils import Conv2dNormRelu, batch_indexing_channel_first, batch_indexing_channel_last
from .pointconv import PointConvNoSampling, WeightNet
from .csrc import k_nearest_neighbor

class PointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, activation='leaky_relu', k=9):
        super().__init__()
        self.k = k

        self.weight_net = WeightNet(2, [9, 9], activation=activation)
        self.linear = nn.Linear(9 * (in_channels + 2), out_channels)

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm1d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm1d(out_channels)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation is None:
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, xy_offset, knn_features, knn_indices=None):
        """
        :param xyz: 3D locations of points, [batch_size, 3, n_points]
        :param features: features of points, [batch_size, in_channels, n_points]
        :param knn_indices: optional pre-computed knn indices, [batch_size, n_points, k]
        :return weighted_features: features of sampled points, [batch_size, out_channels, n_samples]
        """
        batch_size, n_points = xy_offset.shape[0], xy_offset.shape[2]
        knn_features = torch.cat([xy_offset, knn_features], dim=1)  # [bs, in_channels + 2, n_points]
        knn_features_cl = knn_features.permute(0, 2, 3, 1) # [bs, n_points, n_channels + 2]

        # Calculate weights
        weights = self.weight_net(xy_offset)  # [bs, n_weights, n_points, k]

        # Calculate weighted features
        weights = weights.transpose(1, 2)  # [bs, n_points, n_weights, k]
        weighted_features = torch.matmul(weights, knn_features_cl)  # [bs, n_points, n_weights, 2 + in_channels]
        weighted_features = weighted_features.view(batch_size, n_points, -1)  # [bs, n_points, (2 + in_channels) * n_weights]
        weighted_features = self.linear(weighted_features).float()  # [bs, n_points, out_channels]
        weighted_features = self.activation_fn(self.norm_fn(weighted_features.transpose(1, 2)))  # [bs, out_channels, n_points]

        return weighted_features




class FlowEstimator2D(nn.Module):
    def __init__(self, feat_ch, corr_ch, conf_ch, flow_ch, norm=None, conv_last=False):
        super().__init__()

        self.corr_modulator = nn.Sequential(
            Conv2dNormRelu(feat_ch + corr_ch + conf_ch + flow_ch, 2 * corr_ch, kernel_size=3, padding=1, norm=norm),
            Conv2dNormRelu(2 * corr_ch, 2 * corr_ch, kernel_size=3, padding=1, norm=norm),
            )
        self.corr_ch = corr_ch
        self.flow_feat_dim = 32

        self.flow_encoder = Conv2dNormRelu(corr_ch, self.flow_feat_dim, kernel_size=1, padding=0, norm=norm)

        if conv_last:
            self.conv_last = nn.Conv2d(self.flow_feat_dim, 2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_last = None

    def forward(self, feat, corr, feat_conf, last_flow):
        tmp = torch.cat([feat, corr, feat_conf, last_flow], dim=1)
        corr_mod = self.corr_modulator(tmp)
        corr_mul = 2 * torch.sigmoid(corr_mod[:, :self.corr_ch, :, :])
        corr_add = corr_mod[:, self.corr_ch:, :, :]
        corr_new = corr_mul * corr + corr_add
        flow_feat = self.flow_encoder(corr_new)

        if self.conv_last is not None:
            flow = self.conv_last(flow_feat)
            return flow_feat, flow
        else:
            return flow_feat