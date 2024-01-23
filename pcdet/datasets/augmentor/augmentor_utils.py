import numpy as np
import numba
import math
import copy
from ...utils import common_utils
from ...utils import box_utils
from ...ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu

def random_flip_along_x(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
        
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def global_rotation_along_z(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points

def global_rotation_along_x(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_x(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_x(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_x(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points

def global_rotation_along_y(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_y(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_y(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_y(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, return_scale=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:] *= noise_scale
        
    if return_scale:
        return gt_boxes, points, noise_scale
    return gt_boxes, points

def global_scaling_with_roi_boxes(gt_boxes, roi_boxes, points, scale_range, return_scale=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    roi_boxes[:,:, [0,1,2,3,4,5,7,8]] *= noise_scale
    if return_scale:
        return gt_boxes,roi_boxes, points, noise_scale
    return gt_boxes, roi_boxes, points


def random_image_flip_horizontal(image, depth_map, gt_boxes, calib):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        depth_map: (H_depth, W_depth), Depth map
        gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
        calib: calibration.Calibration, Calibration object
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_depth_map: (H_depth, W_depth), Augmented depth map
        aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)
        aug_depth_map = np.fliplr(depth_map)
        
        # Flip 3D gt_boxes by flipping the centroids in image space
        aug_gt_boxes = copy.copy(gt_boxes)
        locations = aug_gt_boxes[:, :3]
        img_pts, img_depth = calib.lidar_to_img(locations)
        W = image.shape[1]
        img_pts[:, 0] = W - img_pts[:, 0]
        pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
        pts_lidar = calib.rect_to_lidar(pts_rect)
        aug_gt_boxes[:, :3] = pts_lidar
        aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]

    else:
        aug_image = image
        aug_depth_map = depth_map
        aug_gt_boxes = gt_boxes

    return aug_image, aug_depth_map, aug_gt_boxes


def random_local_translation_along_x(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 0] += offset
        
        gt_boxes[idx, 0] += offset
    
        # if gt_boxes.shape[1] > 7:
        #     gt_boxes[idx, 7] += offset
    
    return gt_boxes, points


def random_local_translation_along_y(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 1] += offset
        
        gt_boxes[idx, 1] += offset
    
        # if gt_boxes.shape[1] > 8:
        #     gt_boxes[idx, 8] += offset
    
    return gt_boxes, points


def random_local_translation_along_z(gt_boxes, points, offset_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        offset_range: [min max]]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        # augs[f'object_{idx}'] = offset
        points_in_box, mask = get_points_in_box(points, box)
        points[mask, 2] += offset
        
        gt_boxes[idx, 2] += offset
    
    return gt_boxes, points


def global_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    # threshold = max - length * uniform(0 ~ 0.2)
    threshold = np.max(points[:, 2]) - intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    
    points = points[points[:, 2] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] < threshold]
    return gt_boxes, points


def global_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 2]) + intensity * (np.max(points[:, 2]) - np.min(points[:, 2]))
    points = points[points[:, 2] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 2] > threshold]
    
    return gt_boxes, points


def global_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.max(points[:, 1]) - intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] < threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] < threshold]
    
    return gt_boxes, points


def global_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    
    threshold = np.min(points[:, 1]) + intensity * (np.max(points[:, 1]) - np.min(points[:, 1]))
    points = points[points[:, 1] > threshold]
    gt_boxes = gt_boxes[gt_boxes[:, 1] > threshold]
    
    return gt_boxes, points


def local_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
        # augs[f'object_{idx}'] = noise_scale
        points_in_box, mask = get_points_in_box(points, box)
        
        # tranlation to axis center
        points[mask, 0] -= box[0]
        points[mask, 1] -= box[1]
        points[mask, 2] -= box[2]
        
        # apply scaling
        points[mask, :3] *= noise_scale
        
        # tranlation back to original position
        points[mask, 0] += box[0]
        points[mask, 1] += box[1]
        points[mask, 2] += box[2]
        
        gt_boxes[idx, 3:6] *= noise_scale
    return gt_boxes, points


def local_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    # augs = {}
    for idx, box in enumerate(gt_boxes):
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
        # augs[f'object_{idx}'] = noise_rotation
        points_in_box, mask = get_points_in_box(points, box)
        
        centroid_x = box[0]
        centroid_y = box[1]
        centroid_z = box[2]
        
        # tranlation to axis center
        points[mask, 0] -= centroid_x
        points[mask, 1] -= centroid_y
        points[mask, 2] -= centroid_z
        box[0] -= centroid_x
        box[1] -= centroid_y
        box[2] -= centroid_z
        
        # apply rotation
        points[mask, :] = common_utils.rotate_points_along_z(points[np.newaxis, mask, :], np.array([noise_rotation]))[0]
        box[0:3] = common_utils.rotate_points_along_z(box[np.newaxis, np.newaxis, 0:3], np.array([noise_rotation]))[0][0]
        
        # tranlation back to original position
        points[mask, 0] += centroid_x
        points[mask, 1] += centroid_y
        points[mask, 2] += centroid_z
        box[0] += centroid_x
        box[1] += centroid_y
        box[2] += centroid_z
        
        gt_boxes[idx, 6] += noise_rotation
        if gt_boxes.shape[1] > 8:
            gt_boxes[idx, 7:9] = common_utils.rotate_points_along_z(
                np.hstack((gt_boxes[idx, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
                np.array([noise_rotation])
            )[0][:, 0:2]
    
    return gt_boxes, points


def local_frustum_dropout_top(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z + dz / 2) - intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_bottom(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (z - dz / 2) + intensity * dz
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 2] <= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_left(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y + dy / 2) - intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] >= threshold))]
    
    return gt_boxes, points


def local_frustum_dropout_right(gt_boxes, points, intensity_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading, [vx], [vy]],
        points: (M, 3 + C),
        intensity: [min, max]
    Returns:
    """
    for idx, box in enumerate(gt_boxes):
        x, y, z, dx, dy, dz = box[0], box[1], box[2], box[3], box[4], box[5]
        
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        points_in_box, mask = get_points_in_box(points, box)
        threshold = (y - dy / 2) + intensity * dy
        
        points = points[np.logical_not(np.logical_and(mask, points[:, 1] <= threshold))]
    
    return gt_boxes, points


@numba.njit
def get_points_in_box(points, gt_box):
    x, y, z = points[:,0], points[:,1], points[:,2]
    cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
    dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
    shift_x, shift_y, shift_z = x - cx, y - cy, z - cz

    MARGIN = 1e-1
    cosa, sina = math.cos(-rz), math.sin(-rz)
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa

    mask = np.logical_and(np.abs(shift_z) <= dz / 2.0, \
             np.logical_and(np.abs(local_x) <= dx / 2.0 + MARGIN, \
                 np.abs(local_y) <= dy / 2.0 + MARGIN ))

    points = points[mask]

    return points, mask


def get_pyramids(boxes):
    pyramid_orders = np.array([
        [0, 1, 5, 4],
        [4, 5, 6, 7],
        [7, 6, 2, 3],
        [3, 2, 1, 0],
        [1, 2, 6, 5],
        [0, 4, 7, 3]
    ])
    boxes_corners = box_utils.boxes_to_corners_3d(boxes).reshape(-1, 24)
    
    pyramid_list = []
    for order in pyramid_orders:
        # frustum polygon: 5 corners, 5 surfaces
        pyramid = np.concatenate((
            boxes[:, 0:3],
            boxes_corners[:, 3 * order[0]: 3 * order[0] + 3],
            boxes_corners[:, 3 * order[1]: 3 * order[1] + 3],
            boxes_corners[:, 3 * order[2]: 3 * order[2] + 3],
            boxes_corners[:, 3 * order[3]: 3 * order[3] + 3]), axis=1)
        pyramid_list.append(pyramid[:, None, :])
    pyramids = np.concatenate(pyramid_list, axis=1)  # [N, 6, 15], 15=5*3
    return pyramids


def one_hot(x, num_class=1):
    if num_class is None:
        num_class = 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx


def points_in_pyramids_mask(points, pyramids):
    pyramids = pyramids.reshape(-1, 5, 3)
    flags = np.zeros((points.shape[0], pyramids.shape[0]), dtype=np.bool)
    for i, pyramid in enumerate(pyramids):
        flags[:, i] = np.logical_or(flags[:, i], box_utils.in_hull(points[:, 0:3], pyramid))
    return flags


def local_pyramid_dropout(gt_boxes, points, dropout_prob, pyramids=None):
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    drop_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
    drop_pyramid_one_hot = one_hot(drop_pyramid_indices, num_class=6)
    drop_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= dropout_prob
    if np.sum(drop_box_mask) != 0:
        drop_pyramid_mask = (np.tile(drop_box_mask[:, None], [1, 6]) * drop_pyramid_one_hot) > 0
        drop_pyramids = pyramids[drop_pyramid_mask]
        point_masks = points_in_pyramids_mask(points, drop_pyramids)
        points = points[np.logical_not(point_masks.any(-1))]
    # print(drop_box_mask)
    pyramids = pyramids[np.logical_not(drop_box_mask)]
    return gt_boxes, points, pyramids


def local_pyramid_sparsify(gt_boxes, points, prob, max_num_pts, pyramids=None):
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    if pyramids.shape[0] > 0:
        sparsity_prob, sparsity_num = prob, max_num_pts
        sparsify_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
        sparsify_pyramid_one_hot = one_hot(sparsify_pyramid_indices, num_class=6)
        sparsify_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= sparsity_prob
        sparsify_pyramid_mask = (np.tile(sparsify_box_mask[:, None], [1, 6]) * sparsify_pyramid_one_hot) > 0
        # print(sparsify_box_mask)
        
        pyramid_sampled = pyramids[sparsify_pyramid_mask]  # (-1,6,5,3)[(num_sample,6)]
        # print(pyramid_sampled.shape)
        pyramid_sampled_point_masks = points_in_pyramids_mask(points, pyramid_sampled)
        pyramid_sampled_points_num = pyramid_sampled_point_masks.sum(0)  # the number of points in each surface pyramid
        valid_pyramid_sampled_mask = pyramid_sampled_points_num > sparsity_num  # only much than sparsity_num should be sparse
        
        sparsify_pyramids = pyramid_sampled[valid_pyramid_sampled_mask]
        if sparsify_pyramids.shape[0] > 0:
            point_masks = pyramid_sampled_point_masks[:, valid_pyramid_sampled_mask]
            remain_points = points[
                np.logical_not(point_masks.any(-1))]  # points which outside the down sampling pyramid
            to_sparsify_points = [points[point_masks[:, i]] for i in range(point_masks.shape[1])]
            
            sparsified_points = []
            for sample in to_sparsify_points:
                sampled_indices = np.random.choice(sample.shape[0], size=sparsity_num, replace=False)
                sparsified_points.append(sample[sampled_indices])
            sparsified_points = np.concatenate(sparsified_points, axis=0)
            points = np.concatenate([remain_points, sparsified_points], axis=0)
        pyramids = pyramids[np.logical_not(sparsify_box_mask)]
    return gt_boxes, points, pyramids


def local_pyramid_swap(gt_boxes, points, prob, max_num_pts, pyramids=None):
    def get_points_ratio(points, pyramid):
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        alphas = ((points[:, 0:3] - pyramid[3:6]) * vector_0).sum(-1) / np.power(vector_0, 2).sum()
        betas = ((points[:, 0:3] - pyramid[3:6]) * vector_1).sum(-1) / np.power(vector_1, 2).sum()
        gammas = ((points[:, 0:3] - surface_center) * vector_2).sum(-1) / np.power(vector_2, 2).sum()
        return [alphas, betas, gammas]
    
    def recover_points_by_ratio(points_ratio, pyramid):
        alphas, betas, gammas = points_ratio
        surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
        vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
        points = (alphas[:, None] * vector_0 + betas[:, None] * vector_1) + pyramid[3:6] + gammas[:, None] * vector_2
        return points
    
    def recover_points_intensity_by_ratio(points_intensity_ratio, max_intensity, min_intensity):
        return points_intensity_ratio * (max_intensity - min_intensity) + min_intensity
    
    # swap partition
    if pyramids is None:
        pyramids = get_pyramids(gt_boxes).reshape([-1, 6, 5, 3])  # each six surface of boxes: [num_boxes, 6, 15=3*5]
    swap_prob, num_thres = prob, max_num_pts
    swap_pyramid_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= swap_prob
    
    if swap_pyramid_mask.sum() > 0:
        point_masks = points_in_pyramids_mask(points, pyramids)
        point_nums = point_masks.sum(0).reshape(pyramids.shape[0], -1)  # [N, 6]
        non_zero_pyramids_mask = point_nums > num_thres  # ingore dropout pyramids or highly occluded pyramids
        selected_pyramids = non_zero_pyramids_mask * swap_pyramid_mask[:,
                                                     None]  # selected boxes and all their valid pyramids
        # print(selected_pyramids)
        if selected_pyramids.sum() > 0:
            # get to_swap pyramids
            index_i, index_j = np.nonzero(selected_pyramids)
            selected_pyramid_indices = [np.random.choice(index_j[index_i == i]) \
                                            if e and (index_i == i).any() else 0 for i, e in
                                        enumerate(swap_pyramid_mask)]
            selected_pyramids_mask = selected_pyramids * one_hot(selected_pyramid_indices, num_class=6) == 1
            to_swap_pyramids = pyramids[selected_pyramids_mask]
            
            # get swapped pyramids
            index_i, index_j = np.nonzero(selected_pyramids_mask)
            non_zero_pyramids_mask[selected_pyramids_mask] = False
            swapped_index_i = np.array([np.random.choice(np.where(non_zero_pyramids_mask[:, j])[0]) if \
                                            np.where(non_zero_pyramids_mask[:, j])[0].shape[0] > 0 else
                                        index_i[i] for i, j in enumerate(index_j.tolist())])
            swapped_indicies = np.concatenate([swapped_index_i[:, None], index_j[:, None]], axis=1)
            swapped_pyramids = pyramids[
                swapped_indicies[:, 0].astype(np.int32), swapped_indicies[:, 1].astype(np.int32)]
            
            # concat to_swap&swapped pyramids
            swap_pyramids = np.concatenate([to_swap_pyramids, swapped_pyramids], axis=0)
            swap_point_masks = points_in_pyramids_mask(points, swap_pyramids)
            remain_points = points[np.logical_not(swap_point_masks.any(-1))]
            
            # swap pyramids
            points_res = []
            num_swapped_pyramids = swapped_pyramids.shape[0]
            for i in range(num_swapped_pyramids):
                to_swap_pyramid = to_swap_pyramids[i]
                swapped_pyramid = swapped_pyramids[i]
                
                to_swap_points = points[swap_point_masks[:, i]]
                swapped_points = points[swap_point_masks[:, i + num_swapped_pyramids]]
                # for intensity transform
                to_swap_points_intensity_ratio = (to_swap_points[:, -1:] - to_swap_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (to_swap_points[:, -1:].max() - to_swap_points[:, -1:].min()),
                                                     1e-6, 1)
                swapped_points_intensity_ratio = (swapped_points[:, -1:] - swapped_points[:, -1:].min()) / \
                                                 np.clip(
                                                     (swapped_points[:, -1:].max() - swapped_points[:, -1:].min()),
                                                     1e-6, 1)
                
                to_swap_points_ratio = get_points_ratio(to_swap_points, to_swap_pyramid.reshape(15))
                swapped_points_ratio = get_points_ratio(swapped_points, swapped_pyramid.reshape(15))
                new_to_swap_points = recover_points_by_ratio(swapped_points_ratio, to_swap_pyramid.reshape(15))
                new_swapped_points = recover_points_by_ratio(to_swap_points_ratio, swapped_pyramid.reshape(15))
                # for intensity transform
                new_to_swap_points_intensity = recover_points_intensity_by_ratio(
                    swapped_points_intensity_ratio, to_swap_points[:, -1:].max(),
                    to_swap_points[:, -1:].min())
                new_swapped_points_intensity = recover_points_intensity_by_ratio(
                    to_swap_points_intensity_ratio, swapped_points[:, -1:].max(),
                    swapped_points[:, -1:].min())
                
                # new_to_swap_points = np.concatenate([new_to_swap_points, swapped_points[:, -1:]], axis=1)
                # new_swapped_points = np.concatenate([new_swapped_points, to_swap_points[:, -1:]], axis=1)
                
                new_to_swap_points = np.concatenate([new_to_swap_points, new_to_swap_points_intensity], axis=1)
                new_swapped_points = np.concatenate([new_swapped_points, new_swapped_points_intensity], axis=1)
                
                points_res.append(new_to_swap_points)
                points_res.append(new_swapped_points)
            
            points_res = np.concatenate(points_res, axis=0)
            points = np.concatenate([remain_points, points_res], axis=0)
    return gt_boxes, points



################################ points sampling aug ###################################

def remove_boxes_with_no_points(points, gt_boxes, min_num):
    keep_idx = []
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        points_in_box, mask = get_points_in_box(points, box)
        if len(points_in_box) >= min_num:
            keep_idx.append(i)
    return np.array(keep_idx, dtype=np.int32)

@numba.njit
def dist_range_points_sampling(points, dist_ranges, keep_rates):
    indices = np.arange(points.shape[0])
    points_dist = np.sqrt((points[:, :3]**2).sum(1))
    near_points_mask = points_dist < dist_ranges[0]
    far_points_mask = points_dist > dist_ranges[1]
    mid_points_mask = np.logical_not(np.logical_or(near_points_mask, far_points_mask))
    
    near_points_indices = indices[near_points_mask]
    mid_points_indices = indices[mid_points_mask]
    far_points_indices = indices[far_points_mask]
    
    near_points_indices = near_points_indices[:int(near_points_indices.shape[0] * keep_rates[0])]
    mid_points_indices = mid_points_indices[:int(mid_points_indices.shape[0] * keep_rates[1])]
    far_points_indices = far_points_indices[:int(far_points_indices.shape[0] * keep_rates[2])]

    points_indices = np.concatenate((near_points_indices, mid_points_indices, far_points_indices), axis=0)
    np.random.shuffle(points_indices)
    points = points[points_indices]

    return points

def points_down_sampling(points, gt_boxes, dist_ranges=[20, 40], keep_ranges=[[0.5, 1.0], [0.7, 1.0], [0.8, 1.0]]):
    keep_rates = [np.random.uniform(keep_range[0], keep_range[1]) for keep_range in keep_ranges]
    np.random.shuffle(points)
    sampling_points = dist_range_points_sampling(points, np.array(dist_ranges), np.array(keep_rates))
    boxes_keep_idxs = remove_boxes_with_no_points(sampling_points, gt_boxes, 5)

    return sampling_points, boxes_keep_idxs

############################### START: For Per-object Augmentation##################################
####################################################################################################

@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin( - angle)
    rot_cos = np.cos( - angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos

#@numba.njit
def points_transform(box_points, centers, loc_transform, rot_transform):
    num_box = centers.shape[0]

    rot_mat_T = np.zeros((num_box, 3, 3), dtype=centers.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)  # angle: rot_transform[i] -> rot_matrix: rot_mat_T[i]
    
    for i in range(num_box):
            box_points[i][:, :3] -= centers[i, :3]
            box_points[i][:, :3] = np.dot(box_points[i][:, :3], rot_mat_T[i])
            box_points[i][:, :3] += centers[i, :3]
            box_points[i][:, :3] += loc_transform[i]

    return np.concatenate(box_points, axis=0)

@numba.njit
def box3d_transform_(boxes, points, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    new_point_masks = []
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]
            _, mask = get_points_in_box(points, boxes[i])
            new_point_masks.append(mask)
    return new_point_masks

def _select_transform(transform, indices):
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:  # eles: result[i] = 0 -> no noise performed
            result[i] = transform[i, indices[i]]
    return result


#@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises, eps=1e-6):
    '''
        This func aims to perform loc and rot noise no bev gt boxes, each gt box has 100 times chance to perform
        the randomly generated noice on its corresponding box; if a box fails after max times, no noise performed
        and return the original box.
         - boxes: [N, 7], [x, y, z, l, w, h, r],
         - valid_mask: [N],
         - loc_noises: [N, M, 3], M=num_try,
         - rot_noises: [N, M].
    '''

    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]   # num_try: 100
    current_box = np.zeros((1, 7), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)  # default: -1
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:  # only perform noise on object of targeted class, but un-targeted objects are considered into collison test.
            for j in range(num_tests):
                current_box = boxes[i:i+1].copy()
                current_box[:, :3] += loc_noises[i, j]
                current_box[:, 6] += rot_noises[i, j]
                coll_mat = boxes_bev_iou_cpu(current_box, boxes)  # [1, num_valid_box]
                coll_mat[0, i] = 0
                coll_mat = coll_mat == 0
                if coll_mat.all():                # no collision; if loop ends without break -> no noise performed."""
                    success_mask[i] = j               # index of selected noise
                    boxes[i:i+1] = current_box  # i-th box updated
                    break
    return success_mask

@numba.njit
def points_in_boxes(points, boxes):
    box_points = []
    masks = []
    for box in boxes:
        tmp_points, tmp_masks = get_points_in_box(points, box)
        box_points.append(tmp_points)
        masks.append(tmp_masks)
    return box_points, masks

def noise_per_object_v4(gt_boxes, gt_names, points=None, valid_mask=None, rotation_perturb=np.pi / 4, center_noise_std=1.0,\
                         num_try=5, data_aug_with_context=-1.0, data_aug_random_drop=-1.0):
    """
        perform random rotation and translation on each groundtruth independently.

        Args:
            gt_boxes: [N, 7], gt box in lidar, [x, y, z, l, w, h, rx]
            points: [M, 4], point cloud in lidar, all points in the scene.
            rotation_perturb: [-0.785, 0.785],
            center_noise_std: [1.0, 1.0, 0.5],
            global_random_rot_range: [0, 0]
            num_try: 100
    """
    if not valid_mask.any():
        return gt_boxes, points
        
    num_boxes = gt_boxes.shape[0]

    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])

    offset = [0.0, 0.0, 0.0, 0.0, 0.0]
    if data_aug_with_context > 0:   # to enlarge boxes and keep context
        offset = [0.0, 0.0, data_aug_with_context, data_aug_with_context, 0.0]

    # noise on bev_box.
    selected_noise = noise_per_box(gt_boxes.copy(), valid_mask, loc_noises, rot_noises)  #

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    

    ############### remove points in transformerd boxes ###############
    box_points, point_masks = points_in_boxes(points, gt_boxes[valid_mask])
    box_points = copy.deepcopy(box_points)
    point_masks = np.stack(point_masks, axis=0)
    point_masks = np.sum(point_masks, axis=0) == 0
    points = points[point_masks]
    centers = copy.deepcopy(gt_boxes[:, :3][valid_mask])

    ##################sparse and dropout points in objects ###################
    sparse_dropout_points_in_boxes_(box_points, gt_boxes[valid_mask], gt_names[valid_mask],loc_transforms[valid_mask])

    ######################translate and rot box/points #######################
    _ = box3d_transform_(gt_boxes, points, loc_transforms, rot_transforms, valid_mask)
    box_points = points_transform(box_points, centers, loc_transforms[valid_mask], rot_transforms[valid_mask])
    points = np.concatenate([points, box_points], axis=0)

    return gt_boxes, points


################################ object sparse droupout ###################################
def model_func(x):
    x /= 40
    w0, w1, w2, w3, w4, bias = -1.9802, 1.6219, 0.9743, -0.7275, 0.1070, -1.9173 
    y =  w0 * math.log(x) + w1 * x + w2 * math.pow(x, 2) + w3 * math.pow(x, 3) + w4 * math.pow(x, 4) + bias
    return y * 1000

def dist_num_points_func(raw_dist, now_dist, min_num=10):
    raw_num_points = model_func(raw_dist)
    now_num_points = max(model_func(now_dist), min_num)
    return now_num_points / raw_num_points

def dropout_sparse_points(points, gt_box, loc_move, thresh_num=30, min_num=10, times=3):
    sparse_points = points#[sampling_inds]

    ######## points dropout ########
    pyramids = get_pyramids(gt_box[np.newaxis, :])[0]
    point_masks = points_in_pyramids_mask(sparse_points, pyramids)
    point_masks = point_masks.transpose(1, 0)
    pyramids_points_num = np.sum(point_masks, axis=1)
    sampling_flag = False
    if (pyramids_points_num > thresh_num).any():
        while times:
            log_p = pyramids_points_num + 1
            p = log_p / np.sum(log_p)
            remove_pyramids_num = np.random.choice([0, 1, 2], 1, p=[0.25, 0.5, 0.25])
            if remove_pyramids_num == 0:
                break
            remove_pyramids_id = np.random.choice(6, remove_pyramids_num, p=p, replace=False)
            select_points_masks = point_masks[remove_pyramids_id]
            select_points_masks = select_points_masks.transpose(1, 0)
            sampling_points = sparse_points[np.logical_not(select_points_masks.any(1))]
            if sampling_points.shape[0] > min_num:
                sampling_flag = True
                break
            times -= 1

    return sampling_points if sampling_flag else points
    

def sparse_dropout_points_in_boxes_(points, gt_boxes, gt_names, loc_transforms):
    """
        points: N x [Mi x (3 + C)]
        boxes: N x 7
    """
    for i in range(gt_boxes.shape[0]):
        if gt_names[i] != 'Car':
            continue
        points[i] = dropout_sparse_points(points[i], gt_boxes[i], loc_transforms[i])