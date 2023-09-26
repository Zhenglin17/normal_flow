path = '/Users/zhenglin/Documents/normal_flow/scene_03_00_000000'

import numpy as np
import os
import cv2 as cv
from utils import gen_discretized_event_volume
import torch
from utils import gen_event_images
from scipy.spatial.transform import Rotation as R

def load_data(file, format='evimo2v2'):
    if format == 'evimo2v1':
        data = np.load(file, allow_pickle=True)
        meta  = data['meta'].item()
        depth = data['depth']
        mask  = data['mask']
    elif format == 'evimo2v2':
        meta  = np.load(os.path.join(file, 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        depth = np.load(os.path.join(file, 'dataset_depth.npz'))
        mask  = np.load(os.path.join(file, 'dataset_mask.npz'))
    else:
        raise Exception('Unrecognized data format')
    return meta, depth, mask

def load_events(path):
    directory = path
    p = np.load(directory + '/dataset_events_p.npy')
    t = np.load(directory + '/dataset_events_t.npy')
    xy = np.load(directory + '/dataset_events_xy.npy')
    return p, t, xy

class Interval_warp():
    def __init__(self, index, RV_index, events, meta):
        self.index = index
        self.RV_index = RV_index
        self.events = events
        self.meta = meta

    def find_time_stamps(self):
        t_low = self.meta['full_trajectory'][self.index]['cam']['ts']
        t_up = self.meta['full_trajectory'][self.index+1]['cam']['ts']
        return t_low, t_up
    
    def find_events_slices(self):
        t_low, t_up = self.find_time_stamps()
        index_low = np.searchsorted(self.events[:, 2], t_low)
        index_up = np.searchsorted(self.events[:, 2], t_up)
        return self.events[index_low:index_up, :]
    
    def get_pose(self):
        q_1 = self.meta['full_trajectory'][self.index]['cam']['pos']['q']
        q_2 = self.meta['full_trajectory'][self.index+1]['cam']['pos']['q']
        R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],])
        #rotation matrix camera2 to world frame
        R_wc2 = R.from_quat([q_2['x'], q_2['y'], q_2['z'], q_2['w'],])
        R_wc1 = R_wc1.as_matrix()
        R_wc2 = R_wc2.as_matrix()
        T_1 = self.meta['full_trajectory'][self.index]['cam']['pos']['t']
        T_2 = self.meta['full_trajectory'][self.index]['cam']['pos']['t']
        T_wc1 = np.array([T_1['x'], T_1['y'], T_1['z']])
        T_wc2 = np.array([T_2['x'], T_2['y'], T_2['z']])
        R_c1c2 = R_wc1.T @ R_wc2
        return R_c1c2

    def warp_events(self):
        event_slice = self.find_events_slices()
        R_c1c2 = self.get_pose()
        m = self.meta['meta']
        K = np.array([[m['fx'], 0, m['cx']],
                      [0, m['fy'], m['cy']],
                      [0, 0, 1]])
        Coeffs = np.array([m['k1'], m['k2'], m['p1'], m['p2']])
        img_c2 = event_slice[:, 0:2].reshape(1, -1, 2)
        undistort_pts = cv.undistortPoints(img_c2, K, Coeffs)
        space_c2 = np.squeeze(undistort_pts)
        homo_c2 = np.concatenate((space_c2, np.ones((len(undistort_pts), 1))), axis=1)
        space_c1 = (R_c1c2 @ homo_c2.T).T
        img_c1 = cv.projectPoints(space_c1, rvec=np.array([0.0, 0.0, 0.0]), tvec=np.array([0.0, 0.0, 0.0]), cameraMatrix=K, distCoeffs=Coeffs)[0]
        img_c1 = np.squeeze(img_c1)
        events_c1 = np.concatenate((img_c1, event_slice[:, 2:4]), axis=1)
        return events_c1
    
    def get_RV_events(self):
        t_l = self.meta['full_trajectory'][self.RV_index]['cam']['ts']
        t_u = self.meta['full_trajectory'][self.RV_index+1]['cam']['ts']
        index_l = np.searchsorted(self.events[:, 2], t_l)
        index_u = np.searchsorted(self.events[:, 2], t_u)
        return self.events[index_l:index_u, :]
        

if __name__ == '__main__':
    p, t, xy = load_events(path)
    # print(p.shape, t.shape, xy.shape)
    # print(xy[0].max(), xy[1].max())
    # print(xy[0].min(), xy[1].min())
    events = np.concatenate((xy[:, 0].reshape(-1, 1), xy[:, 1].reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1)), axis=1)
    print(events.shape)
    meta, depth, mask = load_data(path, format='evimo2v2')
    warp = Interval_warp(index=5000, RV_index=1000, events=events, meta=meta)
    # t_low, t_up = warp.find_time_stamps()
    # print(t_up-t_low)
    # event_slice = warp.find_events_slices()
    # print(event_slice.shape)
    # R_wc1, R_wc2, T_wc1, T_wc2 = warp.get_pose()
    pts = warp.warp_events()
    index = np.where((pts[:, 0]<640) & (pts[:, 0]>0) & (pts[:, 1]>0) & (pts[:, 1]<480))
    # print(index)
    pts = pts[index]
    print(pts[:, 0].min(), pts[:, 0].max())
    print(pts[:, 1].min(), pts[:, 1].max())
    pts_RV = warp.get_RV_events()
    pts_index = warp.find_events_slices()
    pts[:, [0, 1]] = pts[:, [1, 0]]
    pts_index[:, [0, 1]] = pts_index[:, [1, 0]]
    pts_RV[:, [0, 1]] = pts_RV[:, [1, 0]]
    print(pts_RV.shape, pts_index.shape, pts.shape, pts)
    original = np.concatenate((pts_RV, pts_index), axis=0)
    warp = np.concatenate((pts_RV, pts), axis=0)
    print(pts.shape, pts_index.shape)
    volume_original = gen_discretized_event_volume(events=torch.from_numpy(pts_index), vol_size=[10, 480, 640])
    volume_warp = gen_discretized_event_volume(events=torch.from_numpy(pts_RV
                                                                       ), vol_size=[10, 480, 640])
    image_warp = gen_event_images(volume_warp[None, :, :, :], 'gen')['gen_event_time_image'][0].numpy().sum(0)
    image_original = gen_event_images(volume_original[None, :, :, :], 'gen')['gen_event_time_image'][0].numpy().sum(0)
    # print(image.shape)
    cv.imshow('image', np.concatenate((image_original.T, image_warp.T), axis=1))
    cv.waitKey(0)
    cv.destroyWindow()
