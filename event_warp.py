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
    def __init__(self, index, RV_index, events, meta) -> None:
        self.index = index
        self.RV_index = RV_index
        self.events = events
        self.meta = meta

    def find_time_stamps(self):
        t_low = meta['full_trajectory'][self.index]['cam']['ts']
        t_up = meta['full_trajectory'][self.index+1]['cam']['ts']
        return t_low, t_up
    
    def find_events_slices(self):
        t_low, t_up = self.find_events_slices()
        index_low = np.searchsorted(events[:, 2], t_low)
        index_up = np.searchsorted(events[:, 2], t_up)
        return self.events[index_low:index_up, :]
    
    def get_pose(self):
        q_1 = meta['full_trajectory'][self.idx]['cam']['pos']['q']
        q_2 = meta['full_trajectory'][self.idx+1]['cam']['pos']['q']
        R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],])
        #rotation matrix camera2 to world frame
        R_wc2 = R.from_quat([q_2['x'], q_2['y'], q_2['z'], q_2['w'],])
        R_wc1 = R_wc1.as_matrix()
        R_wc2 = R_wc2.as_matrix()
        T_1 = meta['full_trajectory'][self.idx]['cam']['pos']['t']
        T_2 = meta['full_trajectory'][self.idx]['cam']['pos']['t']
        T_wc1 = np.array([T_1['x'], T_1['y'], T_1['z']])
        T_wc2 = np.array([T_2['x'], T_2['y'], T_2['z']])
        return R_wc1, R_wc2, T_wc1, T_wc2

    def warp_events(self):
        events_slice = self.find_events_slices()
        R_wc1, R_wc2, T_wc1, T_wc2 = self.get_pose()
        

if __name__ == '__main__':
    p, t, xy = load_events(path)
    # print(p.shape, t.shape, xy.shape)
    # print(xy[0].max(), xy[1].max())
    # print(xy[0].min(), xy[1].min())
    events = np.concatenate((xy[:, 1].reshape(-1, 1), xy[:, 0].reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1)), axis=1)
    print(events.shape)
    meta, depth, mask = load_data(path, format='evimo2v2')
    t_low, t_up = find_time_stamps(4000, meta=meta)
    event_slice = find_events_slices(t_low, t_up, events)
    print(event_slice.shape)
    # volume = gen_discretized_event_volume(events=torch.from_numpy(events), vol_size=[10, 480, 640])
    # print(volume.shape)
    # image = gen_event_images(volume[None, :, :, :], 'gen')['gen_event_time_image'][0].numpy().sum(0)
    # print(image.shape)
    # cv.imshow('image', image)
    # cv.waitKey(0)
    # cv.destroyWindow()
