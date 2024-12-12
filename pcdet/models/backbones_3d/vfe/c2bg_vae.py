import torch

from .vfe_template import VFETemplate
from torch import nn as nn
from .c2bg_utils import get_paddings_indicator

# from .vae import VoxelTransformerBloack_HardVoxel
from .vae_2 import VoxelTransformerBloack_HardVoxel
from .C2BFusion import C2BFusion
from .CSFusion import CSFusion
from .CMFusion import CMFusion

from spconv.pytorch.utils import PointToVoxel

class C2BG_MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.mode = 'train' if self.training else 'test'
        self.num_points_in_voxel = model_cfg.MAX_POINTS_PER_VOXEL
        VFE_cfg = model_cfg.VFE[0]
        self.VFE = VFE_cfg.NAME
        self.Fuse_model = model_cfg.FUSE_NAME

        if self.Fuse_model == 'C2BN':
            self.fusion = C2BFusion(pts_channel = 4,
                                    mid_channel = 64,
                                    out_channel = 64)
        elif self.Fuse_model == 'CSN':
            self.fusion = CSFusion(pts_channel = 4,
                                   mid_channel = 64,
                                   out_channel = 64)
        elif self.Fuse_model == 'CMN':
            self.fusion = CMFusion(pts_channel = 4,
                                   mid_channel = 64,
                                   out_channel = 64)
        else:
            self.Fuse_model = None
        
        if self.Fuse_model is not None:
            self.voxel_generator = PointToVoxel(
                vsize_xyz=model_cfg.VOXEL_SIZE, 
                coors_range_xyz=model_cfg.POINT_CLOUD_RANGE, 
                num_point_features=model_cfg.NUM_PTS_FEATS, 
                max_num_voxels=model_cfg.MAX_NUMBER_OF_VOXELS[self.mode], 
                max_num_points_per_voxel=model_cfg.MAX_POINTS_PER_VOXEL,
                device=torch.device("cuda"))

        if self.VFE == 'VAE':
            self.vae = VoxelTransformerBloack_HardVoxel(
                    d_points = VFE_cfg.POINTS_DIM,
                    d_model = VFE_cfg.MODEL_DIM,
                    d_middle = VFE_cfg.MIDDLE_DIM,
                    d_out = VFE_cfg.OUT_DIM,
                    fps_num = VFE_cfg.FPS_NUM,
                )
            self.num_point_features = VFE_cfg.OUT_DIM
        else:
            self.vae = None
            if self.Fuse_model is not None:
                self.num_point_features = 64
        
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        if self.Fuse_model is not None:
            fuse_pts_feats = self.fusion(batch_dict)
            points = batch_dict['points']
            batch_id = points[:, 0]
            voxel_feats_all = []
            indices = []
            voxel_num_points_all = []
            for b in range(batch_dict["batch_size"]):
                fuse_pts_feats_batch = fuse_pts_feats[batch_id==b]
                points_batch = points[batch_id==b]
                voxels_batch, indices_batch, voxel_num_points_batch = self.voxel_generator(torch.concat([points_batch[:, 1:-1], fuse_pts_feats_batch], dim=-1))
                batch_indice = torch.full((voxels_batch.shape[0], 1), b, device=indices_batch.device)
                indices_batch = torch.concat([batch_indice, indices_batch], dim=-1)
                voxel_feats_all.append(voxels_batch)
                indices.append(indices_batch)
                voxel_num_points_all.append(voxel_num_points_batch)
            voxels = torch.vstack(voxel_feats_all)
            voxel_coords = torch.vstack(indices)
            voxel_num_points = torch.cat(voxel_num_points_all)
            batch_dict['voxel_num_points'] = voxel_num_points
            batch_dict['voxel_coords'] = voxel_coords
            voxel_features = voxels[:, :, 3:]
            batch_dict['voxel_features'] = voxel_features
            batch_dict['voxels'] = voxels[:, :, :3]
        else:
            voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']

        if self.vae is not None:
            pv_xyz = batch_dict['voxels'][:, :, :3]
            voxel_coords = batch_dict['voxel_coords']
            batch_size = batch_dict['batch_size']
            batch_index = voxel_coords[:, 0]
            mask = get_paddings_indicator(voxel_num_points, self.num_points_in_voxel, axis=0)
            voxel_features_all = []
            for b in range(batch_size):
                voxel_feats_batch = voxel_features[batch_index == b]
                mask_batch = mask[batch_index == b]
                pv_xyz_batch = pv_xyz[batch_index == b]
                voxel_coords_batch = voxel_coords[batch_index == b]
                voxel_features_batch = self.vae(pv_xyz_batch, voxel_coords_batch[:, 1:], voxel_feats_batch, pv_xyz_batch[mask_batch], mask_batch)
                voxel_features_all.append(voxel_features_batch)
            voxel_features = torch.cat(voxel_features_all)
            batch_dict['voxel_features'] = voxel_features.squeeze(1)
        #for vae_1 no batch dis(may be wrong)
        # if self.vae is not None:
        #     pv_xyz = batch_dict['voxels'][:, :, :3]
        #     voxel_coords = batch_dict['voxel_coords']
        #     mask = get_paddings_indicator(voxel_num_points, self.num_points_in_voxel, axis=0)
        #     voxel_features = self.vae(pv_xyz, voxel_coords[:, 1:], voxel_features, pv_xyz[mask], mask)
        #     batch_dict['voxel_features'] = voxel_features.squeeze(1)

        else:
            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
            points_mean = points_mean / normalizer
            batch_dict['voxel_features'] = points_mean.contiguous()
        return batch_dict

    def transform_points_to_voxels(self, fuse_pts_feats, batch_dict):
        points = batch_dict['points']
        points = torch.concat([points[:, :-1], fuse_pts_feats], dim=-1)
        voxel_output = self.voxel_generator.generate(points)

        return voxel_output


