import argparse
import multiprocessing
from multiprocessing import Process, Pool, pool
import os

import cv2
import numpy as np
import numba as nb
from scipy import signal 
from tqdm import tqdm

from evimo_flow import create_compressed_npz, add_to_npz, close_npz, istarmap

pool.Pool.istarmap = istarmap

def events_to_bucket_covariance(xyt_array, w, h, size, stride, time_scale):
    assert stride == 1

    l_padding = int(np.floor((size - 1) / 2))
    r_padding = int(np.ceil ((size - 1) / 2))
    buckets_w = int(w/stride) + l_padding + r_padding
    buckets_h = int(h/stride) + l_padding + r_padding

    mean_grid = np.zeros((buckets_h, buckets_w, 3), dtype=np.float32)
    count_grid = np.zeros((buckets_h, buckets_w, 1), dtype=np.int32)

    for i in range(xyt_array.shape[0]):
        x, y, t = xyt_array[i, :]

        int_y = int(y)
        y_l = int_y-l_padding
        y_h = int_y+r_padding+1

        int_x = int(x)
        x_l = int_x-l_padding
        x_h = int_x+r_padding+1

        count_grid[y_l:y_h, x_l:x_h] += 1
        mean_grid[y_l:y_h, x_l:x_h, :] += xyt_array[i, :]

    mean_grid /= count_grid

    cov_grid_upper = np.zeros((buckets_h, buckets_w, 6), dtype=np.float32)
    for i in range(xyt_array.shape[0]):
        x, y, t = xyt_array[i, :]

        int_y = int(y)
        y_l = int_y-l_padding
        y_h = int_y+r_padding+1

        int_x = int(x)
        x_l = int_x-l_padding
        x_h = int_x+r_padding+1

        xyt_c = xyt_array[i, :] - mean_grid[y_l:y_h, x_l:x_h, :]

        xyt_c[:, :, 2] *= time_scale

        xx = np.square(xyt_c[:, :, 0])
        xy = xyt_c[:, :, 0] * xyt_c[:, :, 1]
        xt = xyt_c[:, :, 0] * xyt_c[:, :, 2]
        yy = np.square(xyt_c[:, :, 1])
        yt = xyt_c[:, :, 1] * xyt_c[:, :, 2]
        tt = np.square(xyt_c[:, :, 2])

        cov_grid_upper[y_l:y_h, x_l:x_h, :] += np.dstack((xx, xy, xt, yy, yt, tt))

    cov_grid_upper = cov_grid_upper[l_padding:-r_padding, l_padding:-r_padding, :]
    mean_grid      = mean_grid     [l_padding:-r_padding, l_padding:-r_padding, :]
    count_grid     = count_grid    [l_padding:-r_padding, l_padding:-r_padding]

    # Convert to 3x3 fill out bottom half only because eigh only needs lower half by default
    cov_grid = np.empty((cov_grid_upper.shape[0], cov_grid_upper.shape[1], 3, 3), dtype=np.float32)
    cov_grid[:, :, 0, 0] = cov_grid_upper[:, :, 0]
    cov_grid[:, :, 1, 0] = cov_grid_upper[:, :, 1]
    cov_grid[:, :, 2, 0] = cov_grid_upper[:, :, 2]
    cov_grid[:, :, 1, 1] = cov_grid_upper[:, :, 3]
    cov_grid[:, :, 2, 1] = cov_grid_upper[:, :, 4]
    cov_grid[:, :, 2, 2] = cov_grid_upper[:, :, 5]

    return cov_grid, mean_grid, count_grid
events_to_bucket_covariance = nb.jit(nopython = True, cache = True, fastmath=True)(events_to_bucket_covariance)

def buckets_goodness_of_fit(xyt_array, scene_flow, w, h, size, stride):
    l_padding = int(np.floor((size - 1) / 2))
    r_padding = int(np.ceil ((size - 1) / 2))
    buckets_w = int(w/stride) + l_padding + r_padding
    buckets_h = int(h/stride) + l_padding + r_padding

    dot_grid = np.zeros((buckets_h, buckets_w), dtype=np.float32)
    count_grid = np.zeros((buckets_h, buckets_w), dtype=np.int32)

    for i in range(xyt_array.shape[0]):
        x, y, t = xyt_array[i, :]

        int_y = int(y)
        y_l = int_y-l_padding
        y_h = int_y+r_padding+1

        int_x = int(x)
        x_l = int_x-l_padding
        x_h = int_x+r_padding+1

        scene_flow_slice = np.copy(scene_flow[y_l:y_h, x_l:x_h, :])

        scene_flow_slice_2d = scene_flow_slice.reshape(scene_flow_slice.shape[0]*scene_flow_slice.shape[1], 3)
        scene_flow_dot = np.dot(scene_flow_slice_2d, xyt_array[i, :])
        scene_flow_dot = scene_flow_dot.reshape(scene_flow_slice.shape[0], scene_flow_slice.shape[1])

        dot_grid[y_l:y_h, x_l:x_h] += np.abs(scene_flow_dot)
        count_grid[y_l:y_h, x_l:x_h] += 1

    dot_grid   = dot_grid  [l_padding:-r_padding, l_padding:-r_padding]
    count_grid = count_grid[l_padding:-r_padding, l_padding:-r_padding]

    return dot_grid, count_grid
buckets_goodness_of_fit = nb.jit(nopython = True, cache = True, fastmath=True)(buckets_goodness_of_fit)

MIN_EVENTS_IN_BUCKET = 4 # TODO parameter
# TODO no object
class EventPCA:
    def __init__(self, h, w, size, stride, good_thresh):
        self.h = h
        self.w = w
        self.size = size
        self.stride = stride
        self.good_thresh = good_thresh

        self.flow_coor = None
        self.scene_flow = None

    def events_to_covariances_buckets(self, time_scale, xyt_array):
        cov_grid, mean_grid, count_grid = events_to_bucket_covariance(
            xyt_array, self.w, self.h, self.size, self.stride, time_scale)

        # Remove flow vectors with less than 4 events
        elems = cov_grid.shape[0] * cov_grid.shape[1]
        cov_line = cov_grid.reshape([elems, 3, 3])
        mean_line = mean_grid.reshape([elems, 3])
        count_line = count_grid.flatten()

        enough_events = count_line > MIN_EVENTS_IN_BUCKET
        cov_line = cov_line[enough_events, :, :]
        mean_line = mean_line[enough_events, :]
        count_line = count_line[enough_events]

        return cov_line, mean_line, count_line

    def event_scene_flow(self, events_t, events_xy):
        # Needed to improve the numerical calculations
        # Everything being in float32 probably is adding to the sensitivity
        dt = events_t.max() - events_t.min()
        xyt_array = np.hstack((events_xy, events_t.reshape((events_t.shape[0], 1))))

        # TODO parameter
        if dt < 0.000001: # 1 us is less than the timestamp resolution of the sensor
            time_scale = self.size / 0.000001
        else:
            time_scale = self.size / dt

        covs, means, counts = self.events_to_covariances_buckets(time_scale, xyt_array)

        eig_vals, eig_vecs = np.linalg.eigh(covs)

        eig_vecs_index = np.argmin(eig_vals, axis=1)
        self.scene_flow = eig_vecs[np.arange(eig_vecs.shape[0]), :, eig_vecs_index]
        # The time values are mulitplied by this scale,
        # Thus the third column of the covariance matrix is scaled
        # Thus the third component of the eigenvectors is inversely scaled
        # Thus we scale that component again so that time is in seconds
        self.scene_flow[:, 2] *= time_scale

        self.flow_coor = np.rint(means[:, 0:2]).astype(int)

        # Calculate goodness of fit and remove vectors with poor fit
        # than N events
        scene_flow_grid = np.zeros((self.h, self.w, 3), dtype=np.float32)
        scene_flow_grid[self.flow_coor[:, 1], self.flow_coor[:, 0], :] = self.scene_flow
        dot_prod_grid, count_grid = buckets_goodness_of_fit(
            xyt_array, scene_flow_grid, self.w, self.h, self.size, self.stride)
        goodness = dot_prod_grid.flatten()[count_grid.flatten() > MIN_EVENTS_IN_BUCKET]
        good = goodness <= self.good_thresh

        self.scene_flow = self.scene_flow[good, :]
        self.flow_coor = self.flow_coor[good, :]

        return self.flow_coor, self.scene_flow

    def event_normal_flow(self):
        if self.scene_flow is None:
            print("No Scene Flow generated!")
            return

        grad_to_scale = self.scene_flow[:, 0:2]
        grad_to_scale_norm_squared = np.sum(np.square(grad_to_scale), axis=1)

        # TODO don't hardcode 0.2
        grad_to_scale_norm_squared_bad = grad_to_scale_norm_squared < 0.2**2 # Discard small spatial gradients
        grad_to_scale_norm_squared[grad_to_scale_norm_squared_bad] = np.nan
        grad_to_scale_norm_squared[grad_to_scale_norm_squared_bad] = np.nan

        self.nf = -self.scene_flow[:, 2][:, np.newaxis] * (grad_to_scale / grad_to_scale_norm_squared[:, np.newaxis])
        self.nf = np.nan_to_num(self.nf)

        return self.flow_coor, self.nf

def is_sorted(a):
    return np.all(a[:-1] <= a[1:])

def slice_t_by_flow_ts(flow, event_t, event_xy, event_p):
    flow_keys = []
    for f in sorted(flow.files):
        if 'flow_' in f:
            flow_keys.append(f)

    flow_t = flow['t']
    flow_t_end = flow['t_end']

    assert is_sorted(event_t)
    left_indices = np.searchsorted(event_t, flow_t)
    right_indices = np.searchsorted(event_t, flow_t_end, side='right')

    event_t_list = []
    event_xy_list = []
    event_p_list = []
    event_flow_key_list = []
    flow_t_list = []
    flow_t_end_list = []
    for t_idx, flow_key in enumerate(flow_keys):
        start_event_index = left_indices [t_idx]
        end_event_index   = right_indices[t_idx]

        sample_event_t = event_t[start_event_index:end_event_index]
        if sample_event_t.shape[0] > 0:
            event_t_list.append(sample_event_t)
            event_xy_list.append(event_xy[start_event_index:end_event_index, :])
            event_p_list.append(event_p[start_event_index:end_event_index])
            event_flow_key_list.append(flow_key)
            flow_t_list.append(flow_t[t_idx])
            flow_t_end_list.append(flow_t_end[t_idx])

    return event_t_list, event_xy_list, event_p_list, event_flow_key_list, flow_t_list, flow_t_end_list

def slice_t_by_dt(dt, event_t, event_xy):
    t_slices = np.arange(event_t.min(), event_t.max(), dt)
    flow_t = t_slices[:-1]
    flow_t_end = t_slices[1:]

    assert is_sorted(event_t)
    left_indices = np.searchsorted(event_t, flow_t)
    right_indices = np.searchsorted(event_t, flow_t_end, side='right')

    event_t_list = []
    event_xy_list = []
    event_flow_key_list = []
    flow_t_list = []
    flow_t_end_list = []
    for t_idx, t in enumerate(flow_t):
        start_event_index = left_indices [t_idx]
        end_event_index   = right_indices[t_idx]

        sample_event_t = event_t[start_event_index:end_event_index]
        if sample_event_t.shape[0] > 0:
            event_t_list.append(sample_event_t)
            event_xy_list.append(event_xy[start_event_index:end_event_index, :])
            event_flow_key_list.append(None)
            flow_t_list.append(flow_t[t_idx])
            flow_t_end_list.append(flow_t_end[t_idx])

    return event_t_list, event_xy_list, event_flow_key_list, flow_t_list, flow_t_end_list

CONV_X = (np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]) * (1/8)).astype(np.float32)
CONV_Y = (np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]]) * (1/8)).astype(np.float32)
def calculate_E(rgb_image):
    gray_scale = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    dI_dx = signal.convolve2d(gray_scale, CONV_X, 'same')
    dI_dy = signal.convolve2d(gray_scale, CONV_Y, 'same')
    return np.stack((dI_dx, dI_dy), axis=2)

def normal_flow_optical_flow(optical_flow, bgr_image, mask, thresh=0.6, xx_yy=None):
    # Since 3x3 sobel is used, erode mask by 3x3 kernel to
    # get rid of derivatives on edge of field of view
    kernel = np.ones((3,3), dtype=np.uint8)
    mask_dilated = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    delta_E = calculate_E(bgr_image)
    delta_E[mask_dilated == 0] = 0

    norm_E = np.linalg.norm(delta_E, axis=2)

    # Remove sensor noise
    delta_E[norm_E < 10, :] = 0
    norm_E[norm_E < 10] = 0
    delta_E /= 255.0 # Rescale to  0 to 1
    norm_E  /= 255.0 # Rescale to -1 to 1
    # abs_delta_E = np.abs(delta_E)
    # cv2.imshow('grad_E_x', abs_delta_E[:, :, 0] / abs_delta_E[:, :, 0].max())
    # cv2.waitKey(0)

    norm_E_squared = np.square(norm_E)
    norm_E_squared[norm_E_squared == 0] = np.nan
    dI_dt_over_norm_E_squared = np.einsum('ijk,ijk->ij', optical_flow, delta_E) / norm_E_squared
    ground_truth = np.atleast_3d(np.nan_to_num(dI_dt_over_norm_E_squared)) * np.nan_to_num(delta_E)

    # Remove normal flow vectors that are not in the next frame
    if xx_yy is None:
        x = np.arange(0, optical_flow.shape[1], 1)
        y = np.arange(0, optical_flow.shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
    else:
        xx = xx_yy[0]
        yy = xx_yy[1]

    end_locs = ground_truth + np.dstack((xx, yy))
    ground_truth[:, :, 0][end_locs[:, :, 0] < 0] = 0
    ground_truth[:, :, 0][end_locs[:, :, 0] > optical_flow.shape[1]] = 0
    ground_truth[:, :, 1][end_locs[:, :, 1] < 0] = 0
    ground_truth[:, :, 1][end_locs[:, :, 1] > optical_flow.shape[0]] = 0

    mask_dilated[end_locs[:, :, 0] < 0] = 0
    mask_dilated[end_locs[:, :, 0] > optical_flow.shape[1]] = 0
    mask_dilated[end_locs[:, :, 1] < 0] = 0
    mask_dilated[end_locs[:, :, 1] > optical_flow.shape[0]] = 0

    ground_truth_mask = mask_dilated.astype(bool)

    return ground_truth, ground_truth_mask

def event_count_image(exy, shape):
    count = np.zeros((shape[0], shape[1]), dtype=np.float32)
    np.add.at(count, (exy[:, 1], exy[:, 0]), 1.0)
    return count

def event_timestamp_image(et, exy, ep, shape):
    timestamp = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    np.maximum.at(timestamp, (exy[:,1], exy[:,0], 1-ep), et)

    return timestamp

def event_count_image_with_p(exy, ep, shape):
    count_positive = np.zeros((shape[0], shape[1]), dtype=np.float32)
    count_negative = np.zeros((shape[0], shape[1]), dtype=np.float32)
    np.add.at(count_positive, (exy[ep > 0, 1], exy[ep > 0, 0]), 1.0)
    np.add.at(count_negative, (exy[ep < 1, 1], exy[ep < 1, 0]), 1.0)
    return np.dstack((count_positive,count_negative))

def normal_flow_pca(pca, et, exy, shape, dt=None):
    if dt is None:
        dt = et.max() - et.min()

    pca.event_scene_flow(et, exy)
    nf_coor, nf = pca.event_normal_flow()

    nf = nf * dt # PCA gives the velocity, we want the positional delta

    nf_pca = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    nf_pca[nf_coor[:, 1], nf_coor[:, 0], :] = nf
    return nf_pca
    
def generate_normal_flow(sequence, overwrite=False, gen_gt=True, image_shape=None, save_bgr=False):
    nf_file = os.path.join(sequence, 'dataset_normal_flow.npz')

    if not overwrite and os.path.exists(nf_file):
        print('Warning: not overwriting sequence {}'.format(sequence))
        return
    else:
        if gen_gt:
            reproj   = np.load(os.path.join(sequence, 'dataset_reprojected_classical.npz'))
            flow     = np.load(os.path.join(sequence, 'dataset_flow.npz'))
        event_t  = np.load(os.path.join(sequence, 'dataset_events_t.npy'), mmap_mode='r')
        event_xy = np.load(os.path.join(sequence, 'dataset_events_xy.npy'), mmap_mode='r')
        event_p  = np.load(os.path.join(sequence, 'dataset_events_p.npy'), mmap_mode='r')

        if gen_gt and image_shape is None:
            optical_flow = flow[flow.files[0]]
            image_shape = optical_flow.shape
        assert image_shape is not None

        x = np.arange(0, image_shape[1], 1)
        y = np.arange(0, image_shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        xx_yy = (xx, yy)

        nf_npz = create_compressed_npz(nf_file)

        # TODO do not hardcode
        pca = EventPCA(h=image_shape[0],
                       w=image_shape[1],
                       size=4,
                       stride=1,
                       good_thresh=0.1)

        num_samples = 0
        index_to_frame_id = []
        flow_t_list = []
        flow_t_end_list = []
        if gen_gt:
            event_slice_info_lists = slice_t_by_flow_ts(flow, event_t, event_xy, event_p)
            
            for et, exy, ep, flow_key, t, t_end in tqdm(zip(*event_slice_info_lists), total=len(event_slice_info_lists[0])):
                frame_idx = int(flow_key[-10:])
                rjust_frame_idx = str(frame_idx).rjust(10, '0')
                flow_key = 'flow_'                       + rjust_frame_idx
                bgr_key  = 'reprojected_classical_'      + rjust_frame_idx
                mask_key = 'reprojected_classical_mask_' + rjust_frame_idx

                optical_flow = flow[flow_key]
                bgr_image = reproj[bgr_key]
                mask = reproj[mask_key]
                
                if save_bgr:
                    add_to_npz(nf_npz, 'bgr_image_'         + rjust_frame_idx, bgr_image)
                    add_to_npz(nf_npz, 'bgr_image_mask_'    + rjust_frame_idx, mask)

                nf_gt, nf_gt_mask = normal_flow_optical_flow(
                    optical_flow, bgr_image, mask, thresh=0, xx_yy=xx_yy)
                assert nf_gt.dtype == np.float32
                assert nf_gt_mask.dtype == bool

                # Do not add sample if there are no gt vectors in layer
                sum_gt_mask = np.sum(nf_gt_mask)
                nf_gt[nf_gt_mask == False] = 0

                if sum_gt_mask > 0:
                    second_moment = np.sum(np.square(nf_gt)) / sum_gt_mask

                if (sum_gt_mask > 0 and second_moment < 8): # TODO parameter, set based on histogram
                    nf_pca = normal_flow_pca(pca, et, exy, image_shape, dt=(t_end - t))
                    event_count = event_count_image_with_p(exy, ep, image_shape)
                    event_timestamp = event_timestamp_image(et, exy, ep, optical_flow.shape)
                    assert nf_pca.dtype == np.float32
                    assert event_count.dtype == np.float32
                    assert event_timestamp.dtype == np.float32

                    add_to_npz(nf_npz, 'normal_flow_pca_'               + rjust_frame_idx, nf_pca)
                    add_to_npz(nf_npz, 'normal_flow_event_count_'       + rjust_frame_idx, event_count)
                    add_to_npz(nf_npz, 'normal_flow_event_timestamp_'   + rjust_frame_idx, event_timestamp)
                    if gen_gt:
                        add_to_npz(nf_npz, 'normal_flow_gt_'           + rjust_frame_idx, nf_gt)
                        add_to_npz(nf_npz, 'normal_flow_gt_mask_'      + rjust_frame_idx, nf_gt_mask)

                    num_samples += 1
                    index_to_frame_id.append(frame_idx)
                    flow_t_list.append(t)
                    flow_t_end_list.append(t_end)
        else:
            event_slice_info_lists = slice_t_by_dt(0.01, event_t, event_xy)

            nf_pca = normal_flow_pca(pca, et, exy, image_shape, dt=(t_end - t))
            event_count = event_count_image(exy, image_shape)
            event_timestamp = event_timestamp_image(et, exy, ep, optical_flow.shape)
            assert nf_pca.dtype == np.float32
            assert event_count.dtype == np.float32
            assert event_timestamp.dtype == np.float32

            add_to_npz(nf_npz, 'normal_flow_pca_'               + rjust_frame_idx, nf_pca)
            add_to_npz(nf_npz, 'normal_flow_event_count_'       + rjust_frame_idx, event_count)
            add_to_npz(nf_npz, 'normal_flow_event_timestamp_'   + rjust_frame_idx, event_timestamp)
            if gen_gt:
                add_to_npz(nf_npz, 'normal_flow_gt_'           + rjust_frame_idx, nf_gt)
                add_to_npz(nf_npz, 'normal_flow_gt_mask_'      + rjust_frame_idx, nf_gt_mask)

            num_samples += 1
            index_to_frame_id.append(frame_idx)
            flow_t_list.append(t)
            flow_t_end_list.append(t_end)

        add_to_npz(nf_npz, 't', np.array(flow_t_list))
        add_to_npz(nf_npz, 't_end', np.array(flow_t_end_list))
        add_to_npz(nf_npz, 'index_to_frame_id', np.array(index_to_frame_id))
        add_to_npz(nf_npz, 'num_of_samples', np.array(num_samples))
        close_npz(nf_npz)

        assert num_samples > 0
        print(os.path.split(sequence)[-1],
              'normal flow samples', num_samples,
              'optical flow samples', len(event_slice_info_lists[0]),
              'dropped samples', len(event_slice_info_lists[0]) - num_samples)


def generate_normal_flow_list(sequence_list, overwrite=False, gen_gt=True, shape=None, save_bgr=False):
    p_args_list = [(seq, overwrite, gen_gt, shape, save_bgr) for seq in sequence_list]

    cv2.setNumThreads(1)
    with Pool(multiprocessing.cpu_count()) as p:
        list(tqdm(p.istarmap(generate_normal_flow, p_args_list), 
                  total=len(p_args_list), 
                  position=0, 
                  desc='Sequences'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Flows to Image Layers for Autoencoder")
    parser.add_argument('--nogt', action='store_true', default=False)
    parser.add_argument('--save_bgr', action='store_true', default=False)
    parser.add_argument('--resx', type=int, default=None)
    parser.add_argument('--resy', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('sequences', nargs='*',help='sequences to convert')
    args = parser.parse_args()

    if args.resx is not None:
        shape = (args.resy, args.resx)
    else:
        shape = None

    generate_normal_flow_list(args.sequences, args.overwrite, gen_gt=not args.nogt, shape=shape, save_bgr=args.save_bgr)
