from functools import partial
from pcdet.utils import common_utils
import numpy as np
from .c2bg_utils import get_paddings_indicator
import torch
from torch import nn as nn
from torch.nn import functional as F

class C2BFusion(nn.Module):
    def __init__(self,
                 pts_channel,
                 mid_channel,
                 out_channel,
                 img_levels=[0, 1, 2, 3],
                 img_channel=256,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0):
        super().__init__()

        if isinstance(img_levels, int):
            img_levels = [img_levels]
        assert isinstance(img_levels, list)

        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio

        self.img_levels = img_levels
        img_channels_fpn = [img_channel] * len(img_levels)
        mid_channels = [mid_channel] * len(img_levels)

        norm_1d_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        norm_2d_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channel, out_channel),
            norm_1d_layer(out_channel),
        )

        self.lateral_convs_fpn = nn.ModuleList()
        for i in range(len(img_channels_fpn)):
            l_conv = nn.Sequential(
                nn.Conv2d(img_channels_fpn[i], mid_channels[i], 3, padding=1),
                norm_2d_layer(mid_channels[i]),
                nn.ReLU(inplace=False)
            )
            self.lateral_convs_fpn.append(l_conv)

        self.img_transform = nn.Sequential(
            nn.Linear(mid_channel * len(img_levels), out_channel),
            norm_1d_layer(out_channel),
        )

        self.avgpool_1d = nn.AdaptiveAvgPool1d(1)

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channel),
                norm_1d_layer(out_channel),
                nn.ReLU(inplace=False))

        self.sefc_pt = nn.Sequential(
            nn.Linear(out_channel * 2, out_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel, bias=False),
            nn.Sigmoid()
        )

        self.sefc_img = nn.Sequential(
            nn.Linear(out_channel * 2, out_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel, bias=False),
            nn.Sigmoid()
        )

        self.LayerAttention = nn.Sequential(
            nn.Linear(len(img_levels) * mid_channel, (len(img_levels) * mid_channel) // 8),
            nn.ReLU(inplace=True),
            nn.Linear((len(img_levels) * mid_channel) // 8, len(img_levels)),
            nn.Sigmoid()
        )

    def forward(self, batch_dict):
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
        img_feats_fpn = batch_dict['image_fpn']
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
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
            img_feats = img_fpn_feats_ori_shape
            # SE multi-level img features
            for i in range(batch_size):
                img_inter = []
                for level in range(len(self.img_levels)):
                    single_level_inter_feats = F.adaptive_avg_pool2d(img_feats[level][i:i + 1], (1, 1))
                    img_inter.append(single_level_inter_feats.squeeze_().unsqueeze_(0))
                img_inter = torch.cat(img_inter, dim=-1)
                layer_weight = self.LayerAttention(img_inter)
                for level in range(len(self.img_levels)):
                    img_feats[level][i:i + 1] = img_feats[level][i:i + 1].clone() * layer_weight[:, level] + img_feats[
                                                                                                                 level][
                                                                                                             i:i + 1].clone()

            points_feats = self.pts_transform(points_feats)
            img_feats_pre = self.point_sample_nus(batch_dict, voxels_feats, img_feats, voxels_3d, batch_index)
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
            img_feats = img_fpn_feats_ori_shape

            # SE multi-level img features
            for i in range(batch_size):
                img_inter = []
                for level in range(len(self.img_levels)):
                    single_level_inter_feats = F.adaptive_avg_pool2d(img_feats[level][i:i + 1], (1, 1))
                    img_inter.append(single_level_inter_feats.squeeze_().unsqueeze_(0))
                img_inter = torch.cat(img_inter, dim=-1)
                layer_weight = self.LayerAttention(img_inter)
                for level in range(len(self.img_levels)):
                    img_feats[level][i:i + 1] = img_feats[level][i:i + 1].clone() * layer_weight[:, level] + img_feats[level][i:i + 1].clone()

            img_feats = torch.stack(img_feats, dim=1)
            img_feats = img_feats.view(batch_size, -1, h, w)
            calibs = batch_dict['calib']
            img_feats_pre = []
            points_feats = points[:, 1:]
            points_feats = self.pts_transform(points_feats)
            batch_index = points[:, 0]
            for b in range(batch_size):
                calib_batch = calibs[b]
                pts_3d_batch = points[batch_index == b][:, 1:-1]
                pts_feats_batch = points_feats[batch_index == b]
                img_feat_batch = img_feats[b]

                img_pts_batch = self.point_sample_single_kitti(batch_dict, b, pts_feats_batch, img_feat_batch,
                                                               pts_3d_batch, calib_batch)
                img_feats_pre.append(img_pts_batch)

            img_feats_pre = torch.cat(img_feats_pre)
            img_feats_pre = self.img_transform(img_feats_pre)
            pts_pre_fuse = points_feats

        #SE fusion (wrong version?)
        # img_pts_cat = torch.cat([img_feats_pre, pts_pre_fuse], dim=1)
        # fuse_out = (img_feats_pre * self.sefc_pt(img_pts_cat) + img_feats_pre) + (
        #             pts_pre_fuse * self.sefc_img(img_pts_cat) + pts_pre_fuse)

        #SE Fusion (Descirbed in paper)
        img_pool = self.avgpool_1d(img_feats_pre.transpose(1,0)).transpose(1,0)
        pts_pool = self.avgpool_1d(pts_pre_fuse.transpose(1,0)).transpose(1,0)

        cross_pool = torch.concat([img_pool, pts_pool], dim=1)
        fuse_out = (img_feats_pre * self.sefc_pt(cross_pool) + img_feats_pre) + (
            pts_pre_fuse * self.sefc_img(cross_pool) + pts_pre_fuse)
        
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def get_output_feature_dim(self):
        return self.out_channles

    def point_sample_nus(self, batch_dict, voxels_feats, img_feats, voxels_3d, batch_index):
        voxels_3d = voxels_3d[:, self.inv_idx]
        BN, C, H, W = img_feats.size()
        img_feat = img_feats.view(int(BN / 6), 6, C, H, W)

        img_aug_matrix = batch_dict['img_aug_matrix']
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']
        lidar2image = batch_dict['lidar2image']

        batch_size = BN // 6
        # depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)
        img_feats_pre = []
        for b in range(batch_size):
            img_feat_batch = img_feat[b]
            batch_mask = batch_index == b
            cur_coords = voxels_3d[batch_mask][:, 0:3]
            voxels_feats_batch = voxels_feats[batch_mask]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            depth_emb = self.depth_emb(cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # do image aug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            # filter points outside of images
            on_img = (
                    (cur_coords[..., 0] < H)
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < W)
                    & (cur_coords[..., 1] >= 0)
            )
            img_pts = torch.zeros((voxels_feats_batch.shape[0], img_feat_batch.shape[1]), device=img_feat.device)
            for c in range(on_img.shape[0]):
                mask_c = on_img[c]
                masked_coords = cur_coords[c, mask_c].long()
                img_feat_batch_c = img_feat_batch[c]
                image_pts_c = torch.zeros((voxels_feats_batch.shape[0], img_feat_batch_c.shape[0]),
                                          device=img_feat.device)
                image_pts_c[mask_c] = img_feat_batch_c[:, masked_coords[:, 0], masked_coords[:, 1]].permute(1, 0)
                img_pts = img_pts + image_pts_c

            img_pts_depth_emb = F.dropout((img_pts * depth_emb), p=self.dropout_ratio)
            img_feats_pre.append(img_pts_depth_emb)
        img_feats_pre = torch.cat(img_feats_pre)
        return img_feats_pre

    def point_sample_single_kitti(self, batch_dict, batch_index, voxels_feats, img_feats, voxels_3d, calib):
        h, w = batch_dict['images'].shape[2:]

        # inverse pts aug
        if 'noise_scale' in batch_dict:
            voxels_3d[:, :3] /= batch_dict['noise_scale'][batch_index]
        if 'noise_rot' in batch_dict:
            voxels_3d = common_utils.rotate_points_along_z(voxels_3d.unsqueeze(0),
                                                           -batch_dict['noise_rot'][batch_index].unsqueeze(0))
        if 'flip_x' in batch_dict:
            voxels_3d[:, 1] *= -1 if batch_dict['flip_x'][batch_index] else 1
        if 'flip_y' in batch_dict:
            voxels_3d[:, 2] *= -1 if batch_dict['flip_y'][batch_index] else 1
        voxels_2d, _ = calib.lidar_to_img(voxels_3d.squeeze(0).cpu().numpy())
        voxels_2d_int = torch.Tensor(voxels_2d).to(voxels_feats.device).long()
        filter_idx = (0 <= voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0 <= voxels_2d_int[:, 0]) * (
                    voxels_2d_int[:, 0] < w)
        voxels_2d_int = voxels_2d_int[filter_idx]

        # for visual test
        # image = batch_dict['images'].squeeze(0).cpu().numpy().transpose((1, 2, 0))
        # image = np.ascontiguousarray(image)
        # lidar_image_drawer = Local3DVisualizer(image)
        # lidar_image_drawer.draw_projected_pts_on_image(voxels_2d_int, voxels_3d[:,2], self.point_cloud_range[-1])

        image_pts = torch.zeros((voxels_feats.shape[0], img_feats.shape[0]), device=img_feats.device)
        image_pts[filter_idx] = img_feats[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)
        image_pts = F.dropout(image_pts, p=self.dropout_ratio)

        return image_pts
    

def write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                    (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()

def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (torch.Tensor | np.ndarray): Points in shape (N, 3)
        proj_mat (torch.Tensor | np.ndarray):
            Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        (torch.Tensor | np.ndarray): Points in image coordinates,
            with shape [N, 2] if `with_depth=False`, else [N, 3].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yields better results
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)

    return point_2d_res