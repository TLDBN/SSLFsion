import logging

import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import copy
import pickle
from pathlib import Path

from skimage import io

from multiprocessing import cpu_count

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti


# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

def nth_repl(s: str, sub: str, repl: str, n: int):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s

def get_calib(sensor: str = 'hdl64'):
    calib_file = Path(__file__).parent.absolute().parent.parent.parent / 'data' / 'dense' / f'calib_{sensor}.txt' 
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


class DenseDataset(DatasetTemplate):

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, **kwargs): #TODO remove kwargs
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.logger = logger
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        self.sensor_type = dataset_cfg.SENSOR_TYPE
        self.signal_type = dataset_cfg.SIGNAL_TYPE

        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'

        if split_dir.exists():
            self.sample_id_list = ['_'.join(x.strip().split(',')) for x in open(split_dir).readlines()]
        else:
            self.sample_id_list = None

        self.dense_infos = []
        self.include_dense_data(self.mode)

        self.lidar_folder = f'lidar_{self.sensor_type}_{self.signal_type}'
        self.empty_annos = 0


    def include_dense_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading DENSE dataset')
        dense_infos = []

        num_skipped_infos = 0
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                dense_infos.extend(infos)

        self.dense_infos.extend(dense_infos[:])

        if self.logger is not None:
            self.logger.info('Total skipped info %s' % num_skipped_infos)
            self.logger.info('Total samples for DENSE dataset: %d' % (len(dense_infos)))


    def set_split(self, split):

        super().__init__(dataset_cfg=self.dataset_cfg,
                         class_names=self.class_names,
                         root_path=self.root_path,
                         training=self.training,
                         logger=self.logger)

        self.split = split
        split_dir = self.root_path / 'ImageSets' / f'{self.split}.txt'
        # split_dir = self.root_path / 'ImageSets' / 'old_splits' /f'{self.split}.txt'
        print(split_dir)
        if split_dir.exists():
            self.sample_id_list = ['_'.join(x.strip().split(',')) for x in open(split_dir).readlines()]
        else:
            self.sample_id_list = None

    def get_lidar(self, idx):
        lidar_file = self.root_path / self.lidar_folder / ('%s.bin' % idx)
        assert lidar_file.exists(), f'{lidar_file} not found'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)

    def get_image_shape(self, idx):
        #TODO
        img_file = self.root_path / 'cam_stereo_left_lut' / ('%s.png' % idx)
        assert img_file.exists(), f'{img_file} not found'
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_image_path(self, idx):
        #TODO
        image_path = self.root_path / 'cam_stereo_left_lut' / ('%s.png' % idx)
        assert image_path.exists(), f'{image_path} not found'
        return image_path

    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_path / 'cam_stereo_left_lut' / ('%s.png' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        # camera_image = Image.open(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_label(self, idx):
        label_file = self.root_path / 'gt_labels' / 'cam_left_labels_TMP' / ('%s.txt' % idx)
        assert label_file.exists(), f'{label_file} not found'
        return object3d_kitti.get_objects_from_label(label_file, dense=True)

    def get_road_plane(self, idx):
        plane_file = self.root_path / 'velodyne_planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def get_weather(self):
        weather = None
        if 'clear' in self.split:
            weather = 'clear'
        elif 'dense_fog' in self.split:
            weather = 'dense_fog'
        elif 'light_fog' in self.split:
            weather = 'light_fog'
        elif 'snow' in self.split:
            weather = 'snow'
        
        return weather

    def get_infos(self, logger, num_workers=cpu_count(), has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        calibration = get_calib(self.sensor_type)

        def process_single_scene(sample_idx, calib=calibration):
            info = {}
            pc_info = {'num_features': 5, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # try:
            #     img_shape = self.get_image_shape(sample_idx)
            # except (SyntaxError, ValueError) as e:
            #     print(f'{e}\n\n{sample_idx} image seems to be broken')
            img_shape = np.array([1024, 1920], dtype=np.int32)
            
            # image_info = {'image_idx': sample_idx, 'image_shape': img_shape}
            image_path = self.get_image_path(sample_idx)
            image_info = {'image_idx': sample_idx, 'image_shape': img_shape, 'image_path': image_path}
            info['image'] = image_info

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:

                try:                                        # to prevent crash from samples which have no annotations

                    obj_list = self.get_label(sample_idx)

                    # Ignore any class not in relevant classes
                    obj_list = [obj for obj in obj_list if obj.cls_type in self.class_names]

                    if len(obj_list) == 0:
                        raise ValueError
                    
                    annotations = {'name':       np.array([obj.cls_type for obj in obj_list]),
                                   'truncated':  np.array([obj.truncation for obj in obj_list]),
                                   'occluded':   np.array([obj.occlusion for obj in obj_list]),
                                   'alpha':      np.array([obj.alpha for obj in obj_list]),
                                   'bbox':       np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0),
                                   'dimensions': np.array([[obj.l, obj.h, obj.w] for obj in obj_list]),
                                   'location':   np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0),
                                   'rotation_y': np.array([obj.ry for obj in obj_list]),
                                   'score':      np.array([obj.score for obj in obj_list]),
                                   'difficulty': np.array([obj.level for obj in obj_list], np.int32),
                                   'weather': self.get_weather()}

                    num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                    num_gt = len(annotations['name'])
                    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                    annotations['index'] = np.array(index, dtype=np.int32)

                    loc = annotations['location'][:num_objects]
                    dims = annotations['dimensions'][:num_objects]
                    rots = annotations['rotation_y'][:num_objects]
                    loc_lidar = calib.rect_to_lidar(loc)
                    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                    loc_lidar[:, 2] += h[:, 0] / 2
                    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar

                    info['annos'] = annotations

                    if count_inside_pts:
                        pts = self.get_lidar(sample_idx)
                        calib = get_calib(self.sensor_type)
                        if self.dataset_cfg.FOV_POINTS_ONLY:
                            pts_rect = calib.lidar_to_rect(pts[:, 0:3])

                            fov_flag = get_fov_flag(pts_rect, info['image']['image_shape'], calib)

                            # sanity check that there is no frame without a single point in the camera field of view left
                            if max(fov_flag) == 0:

                                sample = nth_repl(sample_idx, '_', ',', 2)

                                message = f'stage: {"train" if self.training else "eval"}, split: {self.split}, ' \
                                        f'sample: {sample} does not have any points inside the camera FOV ' \
                                        f'and will be skipped'

                                try:
                                    self.logger.error(message)
                                except AttributeError:
                                    print(message)

                                new_index = np.random.randint(self.__len__())
                                return self.__getitem__(new_index)

                            pts = pts[fov_flag]
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar) #nboxes, 8 corners, 3coords
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                        for k in range(num_objects):
                            flag = box_utils.in_hull(pts[:, 0:3], corners_lidar[k])
                            num_points_in_gt[k] = flag.sum()
                        annotations['num_points_in_gt'] = num_points_in_gt

                        num_zeros = (num_points_in_gt == 0).sum()

                        for _ in range(num_zeros):
                            part = sample_idx.split("_")
                            logger.debug(f'{"_".join(part[0:2])},{part[2]} contains {num_zeros} label(s) '
                                         f'without a single point inside')

                except ValueError:

                    part = sample_idx.split("_")
                    logger.warning(f'{"_".join(part[0:2])},{part[2]} does not contain any relevant LiDAR labels')

                    return None

                except AssertionError as e:

                    # to continue even though there are missing VLP32 frames
                    logger.error(e)

                    return None


            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = list(tqdm(executor.map(process_single_scene, sample_id_list), total=len(sample_id_list)))

        filtered_for_none_infos = [info for info in infos if info]

        if has_label:

            name_counter = {}
            points_counter = {}

            for info in filtered_for_none_infos:

                for i in range(len(info['annos']['name'])):

                    name = info['annos']['name'][i]
                    points = info['annos']['num_points_in_gt'][i]

                    if name in name_counter:
                        name_counter[name] += 1
                    else:
                        name_counter[name] = 1

                    if name in points_counter:
                        points_counter[name] += points
                    else:
                        points_counter[name] = points

            logger.info('')
            logger.info('Class distribution')
            logger.info('==================')
            for key, value in name_counter.items():
                logger.info(f'{key:12s} {value}')

            logger.info('')
            logger.info('Points distribution')
            logger.info('===================')
            for key, value in points_counter.items():
                logger.info(f'{key:12s} {value}')

            logger.info('')
            logger.info('Average # of points')
            logger.info('===================')
            for key, value in points_counter.items():
                logger.info(f'{key:12s} {value/name_counter[key]:.0f}')
            logger.info('')

        return filtered_for_none_infos

    def create_groundtruth_database(self, logger, info_path=None, used_classes=None, split='train'):

        import torch

        database_save_path = Path(self.root_path) / (f'gt_database' if split == 'train' else f'gt_database_{split}')
        db_info_save_path = Path(self.root_path) / f'dense_dbinfos_{split}.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in tqdm(range(len(infos))):
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        logger.info('')
        for k, v in all_db_infos.items():
            logger.info(f'{k:12s} {len(v)}')
        logger.info('')

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dictionary):
            pred_scores = box_dictionary['pred_scores'].cpu().numpy()
            pred_boxes = box_dictionary['pred_boxes'].cpu().numpy()
            pred_labels = box_dictionary['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.dense_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.dense_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=class_names
        )

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.dense_infos) * self.total_epochs

        return len(self.dense_infos)


    def __getitem__(self, index):
        # index = 563                               # this VLP32 index does not have a single point in the camera FOV
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.dense_infos)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        info = copy.deepcopy(self.dense_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = get_calib(self.sensor_type)

        #before_dict = {'points': copy.deepcopy(points)}

        img_shape = info['image']['image_shape']

        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = get_fov_flag(pts_rect, img_shape, calib)

            # sanity check that there is no frame without a single point in the camera field of view left
            if max(fov_flag) == 0:

                sample = nth_repl(sample_idx, '_', ',', 2)

                message = f'stage: {"train" if self.training else "eval"}, split: {self.split}, ' \
                          f'sample: {sample} does not have any points inside the camera FOV ' \
                          f'and will be skipped'

                try:
                    self.logger.error(message)
                except AttributeError:
                    print(message)

                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = drop_info_with_name(annos, name='DontCare') #TODO check commonutils

            if annos is None:
                print(index)
                sys.exit(1)

            if self.dataset_cfg.DROP_EMPTY_ANNOTATIONS:

                num_before = len(annos['name'])
                annos = drop_infos_with_no_points(annos)
                num_after = len(annos['name'])

                num_diff = num_before - num_after

                if num_diff > 0:
                    self.empty_annos += num_diff
                    try:
                        self.logger.debug(f'annotations without points accumulated to {self.empty_annos}')
                    except AttributeError:
                        pass
            
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar_test = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']

            assert np.abs(gt_boxes_lidar_test - gt_boxes_lidar).sum() < 1e-4

            assert gt_names.shape[0] == gt_boxes_lidar.shape[0]

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]
                
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape

        return data_dict


    # def before_gt_boxes(self, data_dict):
    #
    #     if data_dict.get('gt_boxes', None) is not None:
    #         selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
    #         data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
    #         data_dict['gt_names'] = data_dict['gt_names'][selected]
    #         gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
    #         gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
    #         data_dict['gt_boxes'] = gt_boxes
    #
    #     return data_dict['gt_boxes']


def drop_info_with_name(info, name):

    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]

    #We dont have DontCare in infos
    assert len(keep_indices) == info['name'].shape[0]

    try:
        for key in info.keys():

            if key == 'gt_boxes_lidar' or key == 'weather':
                ret_info[key] = info[key]
            else:
                ret_info[key] = info[key][keep_indices]

    except IndexError:
        return None

    return ret_info


def drop_infos_with_no_points(info):

    ret_info = {}

    keep_indices = [i for i, x in enumerate(info['num_points_in_gt']) if x > 0]

    for key in info.keys():
        if key == 'weather':
            ret_info[key] = info[key]
        else: 
            ret_info[key] = info[key][keep_indices]

    return ret_info

def create_dense_infos(dataset_cfg, class_names, data_path, save_path, logger, workers=cpu_count()):

    dataset = DenseDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = dataset_cfg.DATA_SPLIT['train'], dataset_cfg.DATA_SPLIT['test']
    # test_split = 'test_all_FOV3000_25' #TODO

    #train_filename = save_path / ('dense_infos_%s.pkl' % train_split)
    val_filename = save_path / ('dense_infos_%s.pkl' % val_split)
    # test_filename = save_path / f'dense_infos_{test_split}.pkl'

    print('---------------Start to generate data infos---------------')

    #dataset.set_split(train_split)
    #dense_infos_train = dataset.get_infos(logger, num_workers=workers, has_label=True, count_inside_pts=True)
    #with open(train_filename, 'wb') as f:
    #    pickle.dump(dense_infos_train, f)
    #print('Dense info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    dense_infos_val = dataset.get_infos(logger, num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(dense_infos_val, f)
    print('Dense info val file is saved to %s' % val_filename)

    # dataset.set_split(test_split)
    # dense_infos_test = dataset.get_infos(logger, num_workers=workers, has_label=True, count_inside_pts=True)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(dense_infos_test, f)
    # print('Dense info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    #dataset.set_split(train_split)
    #dataset.create_groundtruth_database(logger, info_path=train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[2] == 'create_dense_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        log = common_utils.create_logger(f'{ROOT_DIR / "data" / "dense" / "1st_run.log"}', log_level=logging.INFO)

        dataset_cfg = EasyDict(yaml.load(open(sys.argv[4])))
        create_dense_infos(dataset_cfg=dataset_cfg,
                           class_names=['Car', 'Pedestrian', 'Cyclist'],
                           data_path=ROOT_DIR / 'data' / 'dense',
                           save_path=ROOT_DIR / 'data' / 'dense',
                           logger=log)
