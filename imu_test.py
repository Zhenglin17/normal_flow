import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Scalar last
# https://www.mathworks.com/help/aerotbx/ug/quatlog.html
def log_q(q):
    theta = np.arccos(q[3])
    if np.abs(theta) > 1e-8:
        v = q[0:3] / np.sin(theta)
        w = v * theta * 2.0 # TODO why 2x?
    else:
        w = np.zeros((3,))
    return w

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
    file = '/home/zhenglin/Documents/left_camera/sfm/eval/scene_03_00_000000'
    meta, depth, mask = load_data(file, format='evimo2v2')
    camera_x = []
    camera_y = []
    camera_z = []
    camera_t = []
    print(len(meta['full_trajectory']))
    print(len(meta['imu']['/prophesee/left/imu']))
    # print(meta.keys())
    # print(meta['imu'].keys())
    # print(meta['imu']['/prophesee/left/imu'][0])
    #print(meta['full_trajectory'][0]['cam'])
    for idx in range(len(meta['full_trajectory'])-1):
        q_1 = meta['full_trajectory'][idx]['cam']['pos']['q']
        q_2 = meta['full_trajectory'][idx+1]['cam']['pos']['q']
        #rpy_1 = meta['full_trajectory'][0]['cam']['pos']['rpy']
        #rotation matrix camera1 to world frame
        R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],]) # Scipy uses scalar last format 
        #rotation matrix camera2 to world frame
        R_wc2 = R.from_quat([q_2['x'], q_2['y'], q_2['z'], q_2['w'],])
        #rotation matrix camera2 to camera1

        dt = meta['full_trajectory'][idx+1]['cam']['ts'] - meta['full_trajectory'][idx]['cam']['ts']
        #print(dt)
        R_c1c2 = R_wc1.as_matrix().T @ R_wc2.as_matrix()

        # Body rates that take R from R_wc1 to R_wc2
        # Two ways to do it.
        # w_matrix = logm(R.from_quat(q[1:5]).as_matrix())
        # w = np.array([-w_matrix[1,2], w_matrix[0, 2], -w_matrix[0, 1]]) / dt
        w = log_q(R.from_matrix(R_c1c2).as_quat()) / dt
        camera_x.append(w[0])
        camera_y.append(w[1])
        camera_z.append(w[2])
        camera_t.append( meta['full_trajectory'][idx]['cam']['ts'])
        # It would be better to plot the signals
    t_min = meta['full_trajectory'][0]['cam']['ts']
    t_max = meta['full_trajectory'][len(meta['full_trajectory'])-2]['cam']['ts']
    fx = interp1d(camera_t, camera_x, kind = 'cubic')
    fy = interp1d(camera_t, camera_x, kind = 'cubic')
    fz = interp1d(camera_t, camera_x, kind = 'cubic')
    xnew = np.linspace(t_min, t_max, num=25737, endpoint=True)
    # plt.figure(figsize=(30, 6))
    # plt.xlabel('timestamp')
    # plt.ylabel('agular velocity')
    # plt.plot(camera_t, camera_x , color = 'g', label='camera_x')
    # plt.plot(camera_t, camera_y , color = 'b', label='camera_y')
    # plt.plot(camera_t, camera_z , color = 'r', label='camera_z')
    # plt.legend()
    # plt.savefig('camera_w.png')

    # imu_x = []
    # imu_y = []
    # imu_z = []
    # imu_t = []
    # for idx in range(len(meta['imu']['/prophesee/left/imu'])):
    #     imu_x.append(-meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['x'])
    #     imu_y.append(-meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['y'])
    #     imu_z.append(meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['z'])
    #     imu_t.append(meta['imu']['/prophesee/left/imu'][idx]['ts'])
    # plt.figure(figsize=(30, 6))
    # plt.xlabel('timestamp')
    # plt.ylabel('agular velocity')
    # plt.plot(imu_t, imu_x, color = 'g', label='camera_x')
    # plt.plot(imu_t, imu_y, color = 'b', label='camera_y')
    # plt.plot(imu_t, imu_z, color = 'r', label='camera_z')
    # plt.legend()
    # plt.savefig('imu_w.png')