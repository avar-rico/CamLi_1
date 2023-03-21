'''
the best version now.
Fusion is performed during feature pyramid and correlation.
We have changed the flow update module of scene flow to converge slower,
which keeps the trace of the optical flow branch and performs better in the end.
'''

import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, interpolate
from .pwc2d_core import FeaturePyramid2D, FlowEstimatorDense2D, ContextNetwork2D, FlowEstimator2D
from .pwc3d_core import FeaturePyramid3D, FlowEstimator3D
from .utils import Conv1dNormRelu, Conv2dNormRelu, project_feat_with_nn_corr, grid_sample_wrapper, project_pc2image, \
    mesh_grid
from .utils import backwarp_2d, backwarp_3d, knn_interpolation, convex_upsample
from .csrc import correlation2d, k_nearest_neighbor
from .pwc3d_add_core import Correlation3D
from .update import SceneFlowEstimatorRefine
from .flow_init import SceneFlowEstimatorInit



class PyramidFeatureFuser2D(nn.Module):
    """
    Bi-CLFM (Bidirectional Camera-LiDAR Fusion Module)
    For clarity, we implementation Bi-CLFM with two individual modules: one for 2D->3D,
    and the other for 3D->2D.
    This module is designed for pyramid feature fusion (3D->2D).
    """

    def __init__(self, in_channels_2d, in_channels_3d, norm=None):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d + 3, in_channels_3d, norm=norm),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm=norm),
            Conv2dNormRelu(in_channels_3d, in_channels_3d, norm=norm),
        )
        self.fuse = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)

    def forward(self, xy, feat_2d, feat_3d, nn_proj=None):
        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])

        out = self.mlps(feat_3d_to_2d)
        out = self.fuse(torch.cat([out, feat_2d], dim=1))

        return out


class PyramidFeatureFuser3D(nn.Module):
    """Pyramid feature fusion (2D->3D)"""

    def __init__(self, in_channels_2d, in_channels_3d, norm=None):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm),
            Conv1dNormRelu(in_channels_2d, in_channels_2d, norm=norm),
        )
        self.fuse = Conv1dNormRelu(in_channels_2d + in_channels_3d, in_channels_3d)

    def forward(self, xy, feat_2d, feat_3d):
        with torch.no_grad():
            feat_2d_to_3d = grid_sample_wrapper(feat_2d, xy)

        out = self.mlps(feat_2d_to_3d)
        out = self.fuse(torch.cat([out, feat_3d], dim=1))

        return out


class CorrFeatureFuser2D(nn.Module):
    """Correlation feature fusion (3D->2D)"""

    def __init__(self, in_channels_2d, in_channels_3d):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d + 5, in_channels_3d),
            Conv2dNormRelu(in_channels_3d, in_channels_3d),
            Conv2dNormRelu(in_channels_3d, in_channels_3d),
        )
        self.fuse = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)

    def forward(self, xy, feat_2d, feat_3d, last_flow_2d, last_flow_3d_to_2d, nn_proj=None):
        feat_3d = torch.cat([feat_3d, last_flow_3d_to_2d], dim=1)

        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])
        feat_3d_to_2d[:, -2:] -= last_flow_2d.detach()

        out = self.mlps(feat_3d_to_2d)
        out = self.fuse(torch.cat([out, feat_2d], dim=1))

        return out


class CorrFeatureFuser3D(nn.Module):
    """Correlation feature fusion (2D->3D)"""

    def __init__(self, in_channels_2d, in_channels_3d):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv1dNormRelu(in_channels_2d + 2, in_channels_2d),
            Conv1dNormRelu(in_channels_2d, in_channels_2d),
            Conv1dNormRelu(in_channels_2d, in_channels_2d),
        )
        self.fuse = Conv1dNormRelu(in_channels_2d + in_channels_3d, in_channels_3d)

    def forward(self, xy, feat_corr_2d, feat_corr_3d, last_flow_3d, last_flow_2d_to_3d):
        with torch.no_grad():
            feat_2d_with_flow = torch.cat([feat_corr_2d, last_flow_2d_to_3d], dim=1)
            feat_2d_to_3d_with_flow = grid_sample_wrapper(feat_2d_with_flow, xy)
            feat_2d_to_3d = feat_2d_to_3d_with_flow
            feat_2d_to_3d[:, -2:] -= last_flow_3d[:, :2]

        out = self.mlps(feat_2d_to_3d)
        out = self.fuse(torch.cat([out, feat_corr_3d], dim=1))

        return out


class DecoderFeatureFuser2D(nn.Module):
    """Decoder feature fusion (3D->2D)"""

    def __init__(self, in_channels_2d, in_channels_3d):
        super().__init__()

        self.mlps = nn.Sequential(
            Conv2dNormRelu(in_channels_3d + 3, in_channels_3d),
            Conv2dNormRelu(in_channels_3d, in_channels_3d),
            Conv2dNormRelu(in_channels_3d, in_channels_3d),
        )
        self.fuse = Conv2dNormRelu(in_channels_2d + in_channels_3d, in_channels_2d)

    def forward(self, xy, feat_2d, feat_3d, nn_proj=None):
        feat_3d_to_2d = project_feat_with_nn_corr(xy, feat_2d, feat_3d, nn_proj[..., 0])

        out = self.mlps(feat_3d_to_2d)
        out = self.fuse(torch.cat([out, feat_2d], dim=1))

        return out


class DecoderFeatureFuser3D(nn.Module):
    """Decoder feature fusion (2D->3D)"""

    def __init__(self, in_channels_2d, in_channels_3d):
        super().__init__()
        self.fuse = Conv1dNormRelu(in_channels_2d + in_channels_3d, in_channels_3d)

    def forward(self, xy, feat_2d, feat_3d):
        with torch.no_grad():
            feat_2d_to_3d = grid_sample_wrapper(feat_2d, xy)
        out = self.fuse(torch.cat([feat_2d_to_3d, feat_3d], dim=1))
        return out


class PWCFusionCore(nn.Module):
    """
    The main architecture of CamLiFlow, which is built on top of PWC-Net and Point-PWC.
    """

    def __init__(self, cfgs2d, cfgs3d, debug=False):
        super().__init__()
        self.cfgs2d, self.cfgs3d, self.debug = cfgs2d, cfgs3d, debug
        corr_channels_2d = (2 * cfgs2d.max_displacement + 1) ** 2

        # PWC-Net 2D (IRR-PWC)
        self.feature_pyramid_2d = FeaturePyramid2D(
            [3, 16, 32, 64, 128, 192],
            norm=cfgs2d.norm.feature_pyramid
        )
        self.feature_aligners_2d = nn.ModuleList([
            nn.Identity(),
            Conv2dNormRelu(32, 64),
            Conv2dNormRelu(64, 64),
            Conv2dNormRelu(128, 64),
            Conv2dNormRelu(192, 64),
        ])
        self.flow_estimator_2d = FlowEstimatorDense2D(
            [64 + corr_channels_2d + 2, 128, 128, 96, 64, 32],
            norm=cfgs2d.norm.flow_estimator,
            conv_last=False,
        )
        # self.flow_estimator_2d = FlowEstimator2D(
        #     [64 + corr_channels_2d + 2, 128, 64, 32],
        #     norm=cfgs2d.norm.flow_estimator,
        #     conv_last=False,
        # )
        self.context_network_2d = ContextNetwork2D(
            [self.flow_estimator_2d.flow_feat_dim + 2, 128, 128, 128, 96, 64, 32],
            dilations=[1, 2, 4, 8, 16, 1],
            norm=cfgs2d.norm.context_network
        )
        self.up_mask_head_2d = nn.Sequential(  # for convex upsampling (see RAFT)
            nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 4 * 9, kernel_size=1, stride=1, padding=0),
        )

        # PWC-Net 3D (Point-PWC)
        self.feature_pyramid_3d = FeaturePyramid3D(
            [32, 64, 128, 192],
            norm=cfgs3d.norm.feature_pyramid,
            k=cfgs3d.k,
        )
        # self.feature_aligners_3d = nn.ModuleList([
        #     nn.Identity(),
        #     Conv1dNormRelu(64, 32),
        #     Conv1dNormRelu(128, 64),
        #     Conv1dNormRelu(192, 64),
        # ])
        # self.correlations_3d = nn.ModuleList([
        #     nn.Identity(),
        #     Correlation3D(32, 32, k=self.cfgs3d.k),
        #     Correlation3D(64, 64, k=self.cfgs3d.k),
        #     Correlation3D(128, 128, k=self.cfgs3d.k),
        #     Correlation3D(192, 192, k=self.cfgs3d.k),
        # ])
        self.correlations_3d = nn.ModuleList([
            nn.Identity(),
            Correlation3D(32, 32, k=32),
            Correlation3D(64, 64, k=32),
            Correlation3D(128, 128, k=32),
            Correlation3D(192, 192, k=32),
        ])

        # Bi-CLFM for pyramid features
        self.pyramid_feat_fusers_2d = nn.ModuleList([
            nn.Identity(),
            PyramidFeatureFuser2D(32, 32, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(64, 64, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(128, 128, norm=cfgs2d.norm.feature_pyramid),
            PyramidFeatureFuser2D(192, 192, norm=cfgs2d.norm.feature_pyramid),
        ])
        self.pyramid_feat_fusers_3d = nn.ModuleList([
            nn.Identity(),
            PyramidFeatureFuser3D(32, 32, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(64, 64, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(128, 128, norm=cfgs3d.norm.feature_pyramid),
            PyramidFeatureFuser3D(192, 192, norm=cfgs3d.norm.feature_pyramid),
        ])

        # Bi-CLFM for correlation features
        self.corr_feat_fusers_2d = nn.ModuleList([
            nn.Identity(),
            CorrFeatureFuser2D(corr_channels_2d, 32),
            CorrFeatureFuser2D(corr_channels_2d, 64),
            CorrFeatureFuser2D(corr_channels_2d, 128),
            CorrFeatureFuser2D(corr_channels_2d, 192),
        ])
        self.corr_feat_fusers_3d = nn.ModuleList([
            nn.Identity(),
            CorrFeatureFuser3D(corr_channels_2d, 32),
            CorrFeatureFuser3D(corr_channels_2d, 64),
            CorrFeatureFuser3D(corr_channels_2d, 128),
            CorrFeatureFuser3D(corr_channels_2d, 192),
        ])

        # self.flow_estimator_3d = nn.ModuleList([
        #     SceneFlowEstimatorRefine(32, 64, 32),
        #     SceneFlowEstimatorRefine(64, 64, 64),
        #     SceneFlowEstimatorRefine(128, 64, 128),
        #     SceneFlowEstimatorInit(192, 192, flow_ch=0),
        # ])
        self.flow_estimator_3d = nn.ModuleList([
            FlowEstimator3D([32 + 64 + 32 + 3, 128, 64], cfgs3d.norm.flow_estimator, k=self.cfgs3d.k),
            FlowEstimator3D([64 + 64 + 64 + 3, 128, 64], cfgs3d.norm.flow_estimator, k=self.cfgs3d.k),
            FlowEstimator3D([128 + 64 + 128 + 3, 128, 64], cfgs3d.norm.flow_estimator, k=self.cfgs3d.k),
            SceneFlowEstimatorInit(192, 192, flow_ch=0),
        ])

        self.estimator_feat_fuser_2d = DecoderFeatureFuser2D(self.flow_estimator_2d.flow_feat_dim, 64)
        # self.estimator_feat_fuser_3d = DecoderFeatureFuser3D(self.flow_estimator_2d.flow_feat_dim, 64)
        self.conv_last_2d = nn.Conv2d(self.flow_estimator_2d.flow_feat_dim, 2, kernel_size=3, stride=1, padding=1)


    def encode(self, image, xyzs):
        feats_2d = self.feature_pyramid_2d(image)
        feats_3d = self.feature_pyramid_3d(xyzs)
        return feats_2d, feats_3d

    def decode(self, xyzs1, xyzs2, feats1_2d, feats2_2d, feats1_3d, feats2_3d, camera_info):
        assert len(xyzs1) + 1 == len(xyzs2) + 1 == len(feats1_2d) == len(feats2_2d) == len(feats1_3d) + 1 == \
               len(feats2_3d) + 1

        flows_2d, flows_3d, flow_feats_2d, flow_feats_3d = [], [], [], []
        for level in range(len(feats1_2d) - 1, 0, -1):
            level_3d = level - 1
            xyz1, feat1_2d, feat1_3d = xyzs1[level_3d], feats1_2d[level], feats1_3d[level_3d]
            xyz2, feat2_2d, feat2_3d = xyzs2[level_3d], feats2_2d[level], feats2_3d[level_3d]

            batch_size, image_h, image_w = feat1_2d.shape[0], feat1_2d.shape[2], feat1_2d.shape[3]
            n_points = xyz1.shape[-1]

            # project point cloud to image
            xy1 = project_pc2image(xyz1, camera_info)
            xy2 = project_pc2image(xyz2, camera_info)

            # sensor coordinate -> image coordinate
            sensor_h, sensor_w = camera_info['sensor_h'], camera_info['sensor_w']
            xy1[:, 0] *= (image_w - 1) / (sensor_w - 1)
            xy1[:, 1] *= (image_h - 1) / (sensor_h - 1)
            xy2[:, 0] *= (image_w - 1) / (sensor_w - 1)
            xy2[:, 1] *= (image_h - 1) / (sensor_h - 1)

            # pre-compute knn indices
            grid = mesh_grid(batch_size, image_h, image_w, xy1.device)  # [B, 2, H, W]
            grid = grid.reshape([batch_size, 2, -1])  # [B, 2, HW]
            nn_proj1 = k_nearest_neighbor(xy1, grid, k=1)  # [B, HW, k]
            nn_proj2 = k_nearest_neighbor(xy2, grid, k=1)  # [B, HW, k]
            knn_1in1 = k_nearest_neighbor(xyz1, xyz1, k=self.cfgs3d.k)

            # fuse pyramid features
            # 2d融合3d投影点的信息
            feat1_2d_fused = self.pyramid_feat_fusers_2d[level](xy1, feat1_2d, feat1_3d, nn_proj1)
            feat2_2d_fused = self.pyramid_feat_fusers_2d[level](xy2, feat2_2d, feat2_3d, nn_proj2)
            # 将2d图像的信息扩展到3d投影点，再融合到3d点的信息中
            feat1_3d_fused = self.pyramid_feat_fusers_3d[level](xy1, feat1_2d, feat1_3d)
            feat2_3d_fused = self.pyramid_feat_fusers_3d[level](xy2, feat2_2d, feat2_3d)
            feat1_2d, feat2_2d = feat1_2d_fused, feat2_2d_fused
            feat1_3d, feat2_3d = feat1_3d_fused, feat2_3d_fused

            if level == len(feats1_2d) - 1:
                last_flow_2d = torch.zeros([batch_size, 2, image_h, image_w], dtype=xy1.dtype, device=xy1.device)
                # last_flow_feat_2d = torch.zeros([batch_size, 32, image_h, image_w], dtype=xy1.dtype, device=xy1.device)
                xyz2_warp, feat2_2d_warp = xyz2, feat2_2d

                feat_corr_3d = self.correlations_3d[level](xyz1, feat1_3d, xyz2_warp, feat2_3d)
                feat_corr_2d = leaky_relu(correlation2d(feat1_2d, feat2_2d_warp, self.cfgs2d.max_displacement), 0.1)

                feat1_2d = self.feature_aligners_2d[level](feat1_2d)

                x_2d = torch.cat([feat_corr_2d, feat1_2d, last_flow_2d], dim=1)
                flow_feat_2d = self.flow_estimator_2d(x_2d)  # [bs, 96, image_h, image_w]

                flow_feat_3d, flow_3d = self.flow_estimator_3d[level_3d](
                        xyz1, feat1_3d, feat_corr_3d)
                flow_2d = self.conv_last_2d(flow_feat_2d)

            else:
                # upsample 2d flow and backwarp
                last_flow_2d = interpolate(flows_2d[-1] * 2, scale_factor=2, mode='bilinear', align_corners=True)
                # last_flow_feat_2d = interpolate(flow_feats_2d[-1], scale_factor=2, mode='bilinear', align_corners=True)
                feat2_2d_warp = backwarp_2d(feat2_2d, last_flow_2d, padding_mode='border')

                # upsample 3d flow and backwarp
                last_flow_3d = knn_interpolation(xyzs1[level_3d + 1], flows_3d[-1], xyz1)
                last_flow_feat_3d = knn_interpolation(xyzs1[level_3d + 1], flow_feats_3d[-1], xyz1)

                xyz2_warp = backwarp_3d(xyz1, xyz2, last_flow_3d)

                feat_corr_3d = self.correlations_3d[level](xyz1, feat1_3d, xyz2_warp, feat2_3d)
                feat_corr_2d = leaky_relu(correlation2d(feat1_2d, feat2_2d_warp, self.cfgs2d.max_displacement), 0.1)

                # fuse correlation features
                # 3D flow对应的是sensor,2D flow对应的是image的尺寸
                last_flow_3d_to_2d = torch.cat([
                    last_flow_3d[:, 0:1] * (image_w - 1) / (sensor_w - 1),
                    last_flow_3d[:, 1:2] * (image_h - 1) / (sensor_h - 1),
                ], dim=1)
                last_flow_2d_to_3d = torch.cat([
                    last_flow_2d[:, 0:1] * (sensor_w - 1) / (image_w - 1),
                    last_flow_2d[:, 1:2] * (sensor_h - 1) / (image_h - 1),
                ], dim=1)
                feat_corr_2d_fused = self.corr_feat_fusers_2d[level](
                    xy1, feat_corr_2d, feat_corr_3d, last_flow_2d, last_flow_3d_to_2d, nn_proj1
                )
                feat_corr_3d_fused = self.corr_feat_fusers_3d[level](
                    xy1, feat_corr_2d, feat_corr_3d, last_flow_3d, last_flow_2d_to_3d
                )
                feat_corr_2d, feat_corr_3d = feat_corr_2d_fused, feat_corr_3d_fused

                feat1_2d = self.feature_aligners_2d[level](feat1_2d)

                # flow decoder (or estimator)
                x_2d = torch.cat([feat_corr_2d, feat1_2d, last_flow_2d], dim=1)
                flow_feat_2d = self.flow_estimator_2d(x_2d)  # [bs, 96, image_h, image_w]

                x_3d = torch.cat([feat_corr_3d, feat1_3d, last_flow_3d, last_flow_feat_3d], dim=1)
                # flow_feat_3d, flow_delta_3d = self.flow_estimator_3d[level_3d](
                #     xyz1, feat1_3d, last_flow_feat_3d, feat_corr_3d, last_flow_3d)
                flow_feat_3d, flow_delta_3d = self.flow_estimator_3d[level_3d](xyz1, x_3d, knn_1in1)

                flow_delta_2d = self.conv_last_2d(flow_feat_2d)

                # residual connection
                flow_2d = last_flow_2d + flow_delta_2d
                flow_3d = last_flow_3d + flow_delta_3d

                flow_feat_2d, flow_delta_2d = self.context_network_2d(torch.cat([flow_feat_2d, flow_2d], dim=1))
                flow_2d = flow_delta_2d + flow_2d

            # save results
            flows_2d.append(flow_2d)
            flows_3d.append(flow_3d)
            flow_feats_2d.append(flow_feat_2d)
            flow_feats_3d.append(flow_feat_3d)

        flows_2d = [f.float() for f in flows_2d][::-1]
        flows_3d = [f.float() for f in flows_3d][::-1]

        # convex upsamling module, from RAFT
        flows_2d[0] = convex_upsample(flows_2d[0], self.up_mask_head_2d(flow_feats_2d[-1]), scale_factor=4)

        for i in range(1, len(flows_2d)):
            flows_2d[i] = interpolate(flows_2d[i] * 4, scale_factor=4, mode='bilinear', align_corners=True)

        return flows_2d, flows_3d
