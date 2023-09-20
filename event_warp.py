path = '/home/zhenglin/Documents/left_camera/sfm/eval/scene_03_00_000000'

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

if __name__ == '__main__':
    p, t, xy = load_events(path)
    print(p.shape, t.shape, xy.shape)
    print(xy[0].max(), xy[1].max())
    print(xy[0].min(), xy[1].min())
    events = np.concatenate((xy[6000:10000, :], t[6000:10000].reshape(-1, 1), p[6000:10000].reshape(-1, 1)), axis=1)
    print(events.shape)
    volume = gen_discretized_event_volume(events=torch.from_numpy(events), vol_size=[1, 10, 480, 640])
    print(volume.shape)
    image = gen_event_images(volume, 'gen')['gen_event_time_image'][0].numpy().sum(0)
    print(image.shape)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyWindow()
