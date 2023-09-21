path = '/Users/zhenglin/Documents/normal_flow/scene_03_00_000000'

import numpy as np
import os
import cv2 as cv
from utils import gen_discretized_event_volume
import torch
from utils import gen_event_images

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

    def find_time_stamps(self):
        t_low = meta['full_trajectory'][self.index]['cam']['ts']
        t_up = meta['full_trajectory'][self.index+1]['cam']['ts']
        return t_low, t_up
    
    def find_events_slices(self):
        t_low, t_up = self.find_events_slices()
        index_low = np.searchsorted(events[:, 2], t_low)
        index_up = np.searchsorted(events[:, 2], t_up)
        return events[index_low:index_up, :]
    
    def relative_pose(self):
        q_1 = meta['full_trajectory'][self.idx]['cam']['pos']['q']
        q_2 = meta['full_trajectory'][idx+1]['cam']['pos']['q']
        RV_pose = ...
        index_pose = ...
        relative_pose = ...

    def warp_events(self):
        events_slice = self.find_events_slices()

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
