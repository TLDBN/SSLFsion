from functools import partial
import spconv.pytorch as spconv
import time
import torch
from torch import nn
from ..backbones_2d.fuser.SSLFusion import SSLFusion
from ...utils.spconv_utils import replace_feature
from .spconv_backbone import post_act_block

class SSLFusionBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        pc_range = model_cfg.PC_RANGE
        voxel_size = model_cfg.VOXEL_SIZE
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1_input_channel = 16
        self.conv1_output_channel = 16
        self.conv2_input_channel = 16
        self.conv2_output_channel = 32
        self.conv3_input_channel = 32
        self.conv3_output_channel = 64
        self.conv4_input_channel = 64
        self.conv4_output_channel = 64

        img_channel = 256
        img_levels = [0, 1, 2, 3]

        self.img_levels = img_levels
        img_channels_fpn = [img_channel] * len(img_levels)
        img_channels_res = [256, 512, 1024, 2048]
        mid_channels = [32, 64, 128, 128]
        norm_2d_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.lateral_convs_fpn = nn.ModuleList()
        for i in range(len(img_channels_fpn)):
            l_conv = nn.Sequential(
                nn.Conv2d(img_channels_fpn[i], mid_channels[i], 3, padding=1),
                norm_2d_layer(mid_channels[i]),
                nn.ReLU(inplace=False)
            )
            self.lateral_convs_fpn.append(l_conv)

        self.lateral_convs_res = nn.ModuleList()

        for i in range(len(img_channels_res)):
            l_conv = nn.Sequential(
                nn.Conv2d(img_channels_res[i], mid_channels[i], 3, padding=1),
                norm_2d_layer(mid_channels[i]),
                nn.ReLU(inplace=False)
            )
            self.lateral_convs_res.append(l_conv)

        self.conv1 = spconv.SparseSequential(
            block(self.conv1_input_channel, self.conv1_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm1'),
        )
        self.fuse_layer1 = SSLFusion(self.conv1_output_channel, mid_channels[0], self.conv2_input_channel, 1, 1, pc_range=pc_range, voxel_size=voxel_size)

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(self.conv2_input_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv2', conv_type='spconv'),
            block(self.conv2_output_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm2'),
            block(self.conv2_output_channel, self.conv2_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm2'),
        )
        self.fuse_layer2 = SSLFusion(self.conv2_output_channel, mid_channels[1], self.conv3_input_channel, 2, 2, pc_range=pc_range, voxel_size=voxel_size)

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(self.conv3_input_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv3', conv_type='spconv'),
            block(self.conv3_output_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm3'),
            block(self.conv3_output_channel, self.conv3_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm3'),
        )
        self.fuse_layer3 = SSLFusion(self.conv3_output_channel, mid_channels[2], self.conv4_input_channel, 4, 3, pc_range=pc_range, voxel_size=voxel_size)

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(self.conv4_input_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                  indice_key='spconv4', conv_type='spconv'),
            block(self.conv4_output_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm4'),
            block(self.conv4_output_channel, self.conv4_output_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='subm4'),
        )
        self.fuse_layer4 = SSLFusion(self.conv4_output_channel, mid_channels[3], self.conv4_input_channel, 8, 4, pc_range=pc_range, voxel_size=voxel_size)

        self.deconv1 = spconv.SparseSequential(
            # [200, 176, 5] <- [400, 352, 11]
            block(self.conv4_output_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, indice_key='spconv4',
                  conv_type='inverseconv'),
            block(self.conv4_input_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm1'),
            block(self.conv4_input_channel, self.conv4_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm1'),
        )
        self.de_fuse_layer1 = SSLFusion(self.conv4_input_channel, mid_channels[2], self.conv3_output_channel, 4, 3, pc_range=pc_range, voxel_size=voxel_size)

        self.deconv2 = spconv.SparseSequential(
            # [400, 352, 11] <- [800, 704, 21]
            block(self.conv3_output_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, indice_key='spconv3',
                  conv_type='inverseconv'),
            block(self.conv3_input_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm2'),
            block(self.conv3_input_channel, self.conv3_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm2'),
        )
        self.de_fuse_layer2 = SSLFusion(self.conv3_input_channel, mid_channels[1], self.conv2_output_channel, 2, 2, pc_range=pc_range, voxel_size=voxel_size)

        self.deconv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [1600, 1408, 41]
            block(self.conv2_output_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, indice_key='spconv2',
                  conv_type='inverseconv'),
            block(self.conv2_input_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm3'),
            block(self.conv2_input_channel, self.conv2_input_channel, 3, norm_fn=norm_fn, padding=1,
                  indice_key='de_subm3'),
        )
        self.de_fuse_layer3 = SSLFusion(self.conv2_input_channel, mid_channels[0], self.conv1_output_channel, 1, 1, pc_range=pc_range, voxel_size=voxel_size)

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        img_feats_fpn = batch_dict['image_fpn']
        # torch.cuda.synchronize()
        # start = time.time()
        if batch_dict['dataset'] == 'NuScenesDataset':
            h, w = batch_dict['camera_imgs'].shape[3:]
            img_fpn_features = [
                lateral_conv(img_feats_fpn[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs_fpn)
            ]
            img_fpn_feats_ori_shape = []
            for i in range(len(self.img_levels)):
                if not img_fpn_features[i].shape[2:] == batch_dict['camera_imgs'].shape[3:]:
                    img_feat_single = nn.functional.interpolate(img_fpn_features[i], (h, w), mode='bilinear',
                                                                align_corners=True)
                    img_fpn_feats_ori_shape.append(img_feat_single)
            batch_dict["img_fpn_feats_ori_shape"] = img_fpn_feats_ori_shape

        else:
            h, w = batch_dict['images'].shape[2:]

            img_fpn_features = [
                lateral_conv(img_feats_fpn[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs_fpn)
            ]
            img_fpn_feats_ori_shape = []
            for i in range(len(self.img_levels)):
                if not img_fpn_features[i].shape == batch_dict['images'].shape:
                    img_feat_single = nn.functional.interpolate(img_fpn_features[i], (h, w), mode='bilinear',
                                                                align_corners=True)
                    img_fpn_feats_ori_shape.append(img_feat_single)
            batch_dict["img_fpn_feats_ori_shape"] = img_fpn_feats_ori_shape
        # end = time.time()
        # print('image feature time:' + str(end - start))
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        # torch.cuda.synchronize()
        # start = time.time()
        # Stage-1,2,3,4
        x_conv1 = self.conv1(x)
        x_fuse1 = self.fuse_layer1(x_conv1, batch_dict, 1)

        x_conv2 = self.conv2(x_fuse1)
        x_fuse2 = self.fuse_layer2(x_conv2, batch_dict, 1)

        x_conv3 = self.conv3(x_fuse2)
        x_fuse3 = self.fuse_layer3(x_conv3, batch_dict, 1)

        x_conv4 = self.conv4(x_fuse3)
        x_fuse4 = self.fuse_layer4(x_conv4, batch_dict, 1)

        # end = time.time()
        # print('conv fuse time:' + str(end - start))
        # Fpn Stage-3,2,1
        x_deconv1 = self.deconv1(x_fuse4)
        x_defuse1 = self.de_fuse_layer1(x_deconv1, batch_dict, 1)
        x_defuse1 = x_defuse1 + x_fuse3

        x_deconv2 = self.deconv2(x_defuse1)
        x_defuse2 = self.de_fuse_layer2(x_deconv2, batch_dict, 1)
        x_defuse2 = x_defuse2 + x_fuse2

        x_deconv3 = self.deconv3(x_defuse2)
        x_defuse3 = self.de_fuse_layer3(x_deconv3, batch_dict, 1)
        x_defuse3 = x_defuse3 + x_fuse1

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_fuse4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })

        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_defuse3,
                'x_conv2': x_defuse2,
                'x_conv3': x_defuse1,
                'x_conv4': x_fuse4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict