# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn import functional as F
from functools import partial
import numpy as np
import time
from pcdet.utils import common_utils
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

class CrossLatentGNN(torch.nn.Module):
    def __init__(self, in_channels, 
                 latent_dim,
                 norm_layer,
                 norm_func):
        super(CrossLatentGNN, self).__init__()

        self.norm_func = norm_func

        # Visible-to-Latent & Latent-to-Visible
        self.modal1_v2l = nn.Sequential(
                        nn.Linear(in_channels, latent_dim),
                        norm_layer(latent_dim),
                        nn.ReLU(inplace=True),
        )
        self.modal2_v2l = nn.Sequential(
                        nn.Linear(in_channels, latent_dim),
                        norm_layer(latent_dim),
                        nn.ReLU(inplace=True),
        )
        self.modal1_l2v = nn.Sequential(
                        nn.Linear(in_channels, latent_dim),
                        norm_layer(latent_dim),
                        nn.ReLU(inplace=True),
        )
        self.modal2_l2v = nn.Sequential(
            nn.Linear(in_channels, latent_dim),
            norm_layer(latent_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, modal1_feats, modal2_feats):
        assert (modal1_feats.shape == modal2_feats.shape)
        N, _ = modal1_feats.shape

        # Generate Modal-Specific Bipartite Graph Adjacency Matrix
        modal1_v2l_graph_adj = self.modal1_v2l(modal1_feats)
        modal2_v2l_graph_adj = self.modal2_v2l(modal2_feats)

        modal1_v2l_graph_adj = self.norm_func(modal1_v2l_graph_adj.view(-1, N), dim=-1)
        modal2_v2l_graph_adj = self.norm_func(modal2_v2l_graph_adj.view(-1, N), dim=-1)

        modal1_l2v_graph_adj = self.modal1_l2v(modal1_feats)
        modal1_l2v_graph_adj = self.norm_func(modal1_l2v_graph_adj.view(-1, N), dim=1)
        modal2_l2v_graph_adj = self.modal2_l2v(modal1_feats)
        modal2_l2v_graph_adj = self.norm_func(modal2_l2v_graph_adj.view(-1, N), dim=1)
        #----------------------------------------------
        # Step1 : Visible-to-Latent 
        #----------------------------------------------
        modal1_latent_node_feature = torch.mm(modal1_v2l_graph_adj, modal1_feats)
        modal2_latent_node_feature = torch.mm(modal2_v2l_graph_adj, modal2_feats)
        cross_modal_latent_node_feature = torch.concat((modal1_latent_node_feature, modal2_latent_node_feature), dim=0)
        #----------------------------------------------
        # Step2 : Latent-to-Latent 
        #----------------------------------------------
        # Generate Dense-connected Graph Adjacency Matrix
        modal1_latent_node_feature_n = self.norm_func(modal1_latent_node_feature, dim=-1)
        modal2_latent_node_feature_n = self.norm_func(modal2_latent_node_feature, dim=-1)
        corss_modal_latent_node_feature_n = torch.concat((modal1_latent_node_feature_n, modal2_latent_node_feature_n), dim=0)
        affinity_matrix = torch.mm(corss_modal_latent_node_feature_n, corss_modal_latent_node_feature_n.permute(1,0))
        affinity_matrix = F.softmax(affinity_matrix, dim=-1)

        cross_modal_latent_node_feature = torch.mm(affinity_matrix, cross_modal_latent_node_feature)
        modal1_latent_node_feature = cross_modal_latent_node_feature[:modal1_latent_node_feature_n.shape[0],:]
        modal2_latent_node_feature = cross_modal_latent_node_feature[modal2_latent_node_feature_n.shape[0]:,:]

        #----------------------------------------------
        # Step3: Latent-to-Visible 
        #----------------------------------------------
        modal1_visible_feature = torch.mm(modal1_latent_node_feature.permute(1, 0), modal1_l2v_graph_adj).view(N, -1)
        modal2_visible_feature = torch.mm(modal2_latent_node_feature.permute(1, 0), modal2_l2v_graph_adj).view(N, -1)
        return modal1_visible_feature + modal1_feats, modal2_visible_feature + modal2_feats

class SSLFusion(nn.Module):
    """SSLFusion.

    Args:
        voxel_channels (int): Channels of voxel features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """
 
    def __init__(self,
                 voxel_channels,
                 mid_channels,
                 out_channels,
                 voxel_stride,
                 voxel_layer,
                 img_levels=[0, 1, 2, 3],
                 img_channels = 256,
                 activate_out=True,
                 fuse_out=True,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=False,
                 lateral_conv=True ,
                 reduction = 16,
                 pc_range=None,
                 voxel_size=None):
        super(SSLFusion, self).__init__()

        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)
        assert pc_range is not None and voxel_size is not None
        self.img_levels = img_levels
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.reduction = reduction
        self.lateral_convs = None
        self.out_channles = mid_channels
        self.lateral_conv = lateral_conv
        self.voxel_stride = voxel_stride
        self.voxel_layer = voxel_layer
        self.point_cloud_range = torch.Tensor(pc_range).cuda().squeeze_()
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()

        norm_1d_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        norm_func = F.normalize

        self.voxel_transform = nn.Sequential(
            nn.Linear(voxel_channels, mid_channels),
            norm_1d_layer(mid_channels),
        )

        self.depth_emb = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.shared_layers = nn.Sequential(
            nn.Linear(mid_channels, 256),
            nn.ReLU(),
            nn.Linear(256, mid_channels),
            norm_1d_layer(mid_channels),
        )

        #Cross-Modal latent GNN
        self.corss_gnn_layer = CrossLatentGNN(mid_channels, 128, norm_layer=norm_1d_layer, norm_func=norm_func)

        # Modality-specific attention layers
        self.attention_lidar = nn.Sequential(
            nn.Linear(mid_channels, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, mid_channels, bias=False),
            nn.Sigmoid(),
        )

        self.attention_image = nn.Sequential(
            nn.Linear(mid_channels, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, mid_channels, bias=False),
            nn.Sigmoid(),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                norm_1d_layer(out_channels),
                nn.ReLU(inplace=True))

    def forward(self, sp_voxel, batch_dict, indice):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """

        batch_size = batch_dict['batch_size']
        if indice == 1:
            img_feats = batch_dict['img_fpn_feats_ori_shape'][self.voxel_layer - 1]
        else:
            img_feats = batch_dict['img_res_feats_ori_shape'][self.voxel_layer - 1]
        batch_index = sp_voxel.indices[:, 0]
        spatial_indices = sp_voxel.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size[self.inv_idx] + self.point_cloud_range[:3][self.inv_idx]
        calibs = batch_dict['calib']
        voxels_feats = sp_voxel.features
        img_feats_pre = []
        voxels_feats = self.voxel_transform(voxels_feats)
        for b in range(batch_size):
            calib_batch = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index==b]
            voxels_feats_batch = voxels_feats[batch_index==b]
            img_feat_batch = img_feats[b]
            
            img_pts_batch = self.point_sample_single_kitti(batch_dict, b, voxels_feats_batch, img_feat_batch, voxels_3d_batch, calib_batch)           
            img_feats_pre.append(img_pts_batch)
            
        img_feats_pre = torch.cat(img_feats_pre)
        
        pts_feats_pre = voxels_feats
        # Shared representation
        shared_lidar = self.shared_layers(pts_feats_pre)
        shared_image = self.shared_layers(img_feats_pre)

        #Cross-Modal Latent GNN Update
        pts_gnn_feats, img_gnn_feats = self.corss_gnn_layer(shared_lidar, shared_image)

        add_feats = shared_lidar + shared_image

        # Cross-modal attention
        att_weights_lidar = self.attention_lidar(add_feats)
        att_weights_image = self.attention_image(add_feats)

        # Weighted features
        weighted_shared_lidar = (pts_gnn_feats * att_weights_lidar) + pts_feats_pre
        weighted_shared_image = (img_gnn_feats * att_weights_image) + img_feats_pre

        fuse_out = weighted_shared_image + weighted_shared_lidar

        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        sp_voxel = sp_voxel.replace_feature(fuse_out)
        torch.cuda.empty_cache()
        return sp_voxel
    

    def get_output_feature_dim(self):
        return self.out_channles


    def point_sample_single_kitti(self, batch_dict, batch_index, voxels_feats, img_feats, voxels_3d, calib):
        h, w = batch_dict['images'].shape[2:]

        #inverse pts aug
        if 'noise_scale' in batch_dict:
            voxels_3d[:, :3] /= batch_dict['noise_scale'][batch_index]
        if 'noise_rot' in batch_dict:
            voxels_3d = common_utils.rotate_points_along_z(voxels_3d[:, self.inv_idx].unsqueeze(0), -batch_dict['noise_rot'][batch_index].unsqueeze(0))[0, :, self.inv_idx]
        if 'flip_x' in batch_dict:
            voxels_3d[:, 1] *= -1 if batch_dict['flip_x'][batch_index] else 1
        if 'flip_y' in batch_dict:
            voxels_3d[:, 2] *= -1 if batch_dict['flip_y'][batch_index] else 1
        voxels_2d, _ = calib.lidar_to_img(voxels_3d[:, self.inv_idx].cpu().numpy())
        voxels_2d_int = torch.Tensor(voxels_2d).to(voxels_feats.device).long()
        filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)
        voxels_2d_int = voxels_2d_int[filter_idx]

        depth_emb = self.depth_emb(voxels_3d)

        image_pts = torch.zeros((voxels_feats.shape[0], img_feats.shape[0]), device=img_feats.device)
        image_pts[filter_idx] = img_feats[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)
        if self.training:
            image_pts = F.dropout(image_pts * depth_emb, p=self.dropout_ratio) 
        else:
            image_pts = image_pts * depth_emb
        return image_pts
