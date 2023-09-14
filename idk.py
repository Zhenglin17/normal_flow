import os
import numpy as np
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

if __name__ == '__main__':
    file = '/Users/zhenglin/Desktop/code/scene_03_00_000000'
    meta, depth, mask = load_data(file, format='evimo2v2')
    # print(meta.keys())
    # print(meta['imu'].keys())
    # print(meta['imu']['/prophesee/left/imu'][0])
    print(len(meta['full_trajectory']))
    print(meta['full_trajectory'][0]['cam'])
    print(len(meta['imu']['/prophesee/left/imu']))
    q_1 = meta['full_trajectory'][0]['cam']['pos']['q']
    q_2 = meta['full_trajectory'][1]['cam']['pos']['q']
    rpy_1 = meta['full_trajectory'][0]['cam']['pos']['rpy']
    #rotation matrix camera1 to world frame
    R_1 = R.from_quat([q_1['w'], q_1['x'], q_1['y'], q_1['z']]) 
    #rotation matrix camera2 to world frame
    R_2 = R.from_quat([q_2['w'], q_2['x'], q_2['y'], q_2['z']])
    #rotation matrix camera2 to camera1
    R_trans = np.dot(R_2.as_matrix(), np.linalg.inv(R_1.as_matrix()))
    print(np.linalg.det(R_trans))
    #rotation matrix camera1 to camera2
    R_trans = R.from_matrix(np.linalg.inv(R_trans))
    #rotation from camera1 to camera2 in axis angle representation, the axis in expressed in camera1 coordinate
    mrp = R_trans.as_rotvec()
    #transfer the axis from camera1 to world frame
    #print(np.dot(R_1.as_matrix(), mrp/0.019744))
    #divided by time interval should be equal to the imu angular velocity
    print(mrp/0.019744)
    print(meta['imu']['/prophesee/left/imu'][0]['angular_velocity'])
    