import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import furthest_point_sample
from functools import partial

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def abs_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point abs distance, [B, N, M]
    Input:
        src: source points, [N, M, C]
        dst: target points, [N, M, C]
    """
    return torch.abs(src[:, :, None] - dst[:, None])

def FPS_HardVoxel(xyz, num_points):
    """
            input: xyz: (n, m, 3), num_points: (k)
            output: idx: (n, k)
            """

    #for mmcv fps version
    xyz = xyz.unsqueeze(0).contiguous()
    assert xyz.is_contiguous()
    idx = furthest_point_sample(xyz, num_points)
    # idx = furthest_point_sample(xyz, num_points)
    return idx

class PointAttention(nn.Module):
    def __init__(self, d_points, d_model) -> None:
        super().__init__()
        
        self.fc_alpha = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_epsilon = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_omega = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(self, pv_xyz, features):
        """
            Args:
                pv_xyz((B X N) x M x 3): N: num of voxels, M: num of points in a voxel, 3:(x, y, z)
                features((B X N) x M x C): points features in voxels
            
            Returns:

        """
           
        dist = abs_distance(pv_xyz, pv_xyz)
        #version 1
        pij = self.fc_delta(dist)

        # pre = self.fc_pre(features)
        pre = features
        q = self.fc_alpha(pv_xyz)
        k = self.fc_beta(pv_xyz)
        v = self.fc_omega(features)

        
        w = self.fc_epsilon((q[:, :, None] - k[:, None] + pij) / np.sqrt(k.size(-1)))
        wij = F.softmax(w, dim=-2)
        res = torch.einsum('bmmf,bmf->bmf', wij, v)
        res = torch.cat([res,pre], dim=-1)

        return res 

class VoxelAttention(nn.Module):
    def __init__(self, d_points, d_model) -> None:
        super().__init__()
        #version 1
        self.fc_alpha = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_epsilon = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_omega = nn.Sequential(
            nn.Linear(d_points, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(self, p_xyz, v_xyz, p_features, v_featrues):
        v_xyz = v_xyz.unsqueeze_(1)
        dist = abs_distance(p_xyz, v_xyz).squeeze_()
        pv = self.fc_delta(dist)


        pre = v_featrues
        q = self.fc_alpha(v_featrues)
        k = self.fc_beta(p_features)
        v = self.fc_omega(p_features)

        w = F.softmax((q @ k.transpose(1, 2) + q @ pv.transpose(1, 2)) / np.sqrt(k.size(-1)), dim = -2)

        res = torch.einsum('bnm,bmf->bnf', w, v)
        res = res + pre
        return res

class GlobalAttention(nn.Module):
    def __init__(self, d_points, d_model) -> None:
        super().__init__()
        self.fc_alpha = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_epsilon = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_omega = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.fc_pre = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def forward(self, p_xyz, features, n_sample):
        idx = FPS_HardVoxel(p_xyz, n_sample).type(torch.LongTensor)
        f = features[idx].squeeze_(0)

        # pre = self.fc_pre(features)
        pre = features
        q = self.fc_alpha(features)
        k = self.fc_beta(f)
        v = self.fc_omega(f)

        w = (q @ k.transpose(0, 1)) /  np.sqrt(k.size(-1))
        res = w @ v

        res = torch.cat([res, pre], dim = -1)
        return res

class VoxelTransformerBloack_HardVoxel(nn.Module):

    def __init__(self, d_points, d_model, d_middle, d_out, fps_num) -> None:
        super().__init__()
        self.n_sample = fps_num
        norm_1d_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.pointattn = PointAttention(d_model, d_middle)
        self.globalattn = GlobalAttention(d_model, d_middle)
        self.voxelattn = VoxelAttention((d_middle + d_model) * 2, 2 * (d_model + d_middle))
        self.fc1 = nn.Linear(d_points, d_middle)
        self.bn1 = norm_1d_layer(d_middle)
        self.bnp = norm_1d_layer(d_middle + d_model)
        self.bng = norm_1d_layer(d_middle + d_model)
        self.bnv = norm_1d_layer(2 * (d_middle + d_model))
        self.bn2 = norm_1d_layer(d_out)
        self.fc3 = nn.Linear(2 * (d_middle + d_model), d_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, p_xyz, v_xyz, featrues, point_coors, mask):
        # version 1 with batch Norm
        pre = featrues
        x = self.relu((self.bn1((self.fc1(featrues)).transpose(1, 2))).transpose(1, 2))
        local_f = self.relu((self.bnp((self.pointattn(p_xyz, x)).transpose(1, 2))).transpose(1, 2))
        gf = self.globalattn(point_coors, x[mask], self.n_sample)
        global_f = featrues.new_zeros(
            size=(featrues.size(0), featrues.size(1),
                  gf.size(-1)))
        global_f[mask] = gf

        global_f = self.relu((self.bng(global_f.transpose(1, 2))).transpose(1, 2))
        x = torch.cat([local_f, global_f], dim = -1)
        if v_xyz is not None:
            v = torch.max(x, dim=1)[0].unsqueeze_(1)
            voxel_f = self.relu((self.bnv((self.voxelattn(p_xyz, v_xyz, x, v)).transpose(1, 2))).transpose(1, 2))

            out = self.relu((self.bn2((self.fc3(voxel_f)).transpose(1, 2))).transpose(1, 2))
            return out
        else:
            return self.relu((self.bn2((self.fc3(x)).transpose(1, 2))).transpose(1, 2))


