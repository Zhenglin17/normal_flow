import numpy as np
from scipy.spatial.transform import Rotation as R
from event_warp import load_data
from event_warp import load_events
from utils import path
import cv2 as cv
from matplotlib import pyplot as plt


path = path()
flow = np.load(path + '/dataset_normal_flow.npz')
p, t, xy = load_events(path)
meta, depth, mask = load_data(path, format='evimo2v2')
# print(flow['normal_flow_pca_0000000001'].shape)
# print(flow['normal_flow_event_count_0000000001'].shape)
# print(flow['normal_flow_event_timestamp_0000000001'].shape)
# print(flow['normal_flow_gt_0000000001'].shape)
# print(flow['normal_flow_gt_mask_0000000001'].shape)
def get_intrinsics():
    m = meta['meta']
    K = np.array([[m['fx'], 0, m['cx']],
                  [0, m['fy'], m['cy']],
                  [0, 0, 1]])
    Coeffs = np.array([m['k1'], m['k2'], m['p1'], m['p2']])
    return K, Coeffs

def generate_rotation_t():
    time_list = []
    for index in range(len(meta['full_trajectory'])):
        time_list.append(meta['full_trajectory'][index]['cam']['ts'])
    time_list = np.array(time_list)
    return time_list

def generate_W_t():
    time_list = []
    for index in range(len(meta['imu']['/prophesee/left/imu'])):
        time_list.append(meta['imu']['/prophesee/left/imu'][index]['ts'])
    time_list = np.array(time_list)
    return time_list

class get_t_depth():
    def __init__(self, frame):
        self.frame = frame

    def get_normal_flow(self):
        normal_flow_gt = []
        for file in flow.files:
            if (('gt' in file) and ('mask' not in file)):
                normal_flow_gt.append(file)
        # print(np.count_nonzero(flow[normal_flow_gt[self.frame]]))
        return flow[normal_flow_gt[self.frame]]
    
    def get_time_stamps(self):
        t = flow['t'][self.frame]
        return t

    def nearest_sample_index_W(self):
        t= self.get_time_stamps()
        R_time = generate_W_t()
        idx = np.searchsorted(R_time, t)
        # print(idx)
        return idx
    
    def nearest_sample_index_rotation(self):
        t= self.get_time_stamps()
        R_time = generate_rotation_t()
        idx = np.searchsorted(R_time, t)
        # print(idx)
        return idx
    
    def get_t(self):
        idx = self.nearest_sample_index_rotation()
        q_1 = meta['full_trajectory'][idx]['cam']['pos']['q']
        R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],])
        R_c1w = R_wc1.as_matrix().T
        T_1 = meta['full_trajectory'][idx]['cam']['pos']['t']
        T_2 = meta['full_trajectory'][idx+1]['cam']['pos']['t']
        dt = meta['full_trajectory'][idx+1]['cam']['ts'] - meta['full_trajectory'][idx]['cam']['ts']
        # print(time)
        v_1 = np.array([T_1['x'], T_1['y'], T_1['z']])
        v_2 = np.array([T_2['x'], T_2['y'], T_2['z']])
        t = np.dot(R_c1w, (v_1-v_2)/dt)
        return t
    
    def get_W(self):
        idx = self.nearest_sample_index_W()
        x = -meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['x']
        y = -meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['y']
        z = meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['z']
        W = [x, y, z]
        W = np.array(W)
        return W
    
    def compute_new_normal(self):
        dt = 0.005
        normal = self.get_normal_flow()
        all_zero = np.logical_and(normal[:, :, 0] == 0,
                                  normal[:, :, 1] == 0)
        all_zero = np.where(all_zero)
        test_array = np.zeros((480, 640))
        test_array[all_zero] = 1
        cord = np.argwhere(test_array == 0)
        idx = np.where(test_array == 0)
        n_flow = normal[idx]
        cord_flow = cord + n_flow * dt
        K, Coeffs = get_intrinsics()
        # print(K, Coeffs)
        cord = cord.astype(np.float64)
        undistort_cord = cv.undistortPoints(cord, K, Coeffs)
        undistort_cord_flow = cv.undistortPoints(cord_flow, K, Coeffs)
        new_n_flow = (undistort_cord_flow - undistort_cord)/dt
        undistort_cord = np.squeeze(undistort_cord)
        new_n_flow = np.squeeze(new_n_flow)
        return cord, undistort_cord, new_n_flow
         
    def cal_flow(self, x, y):
        f = 1
        # print(f)
        w = np.array(self.get_W())
        r = np.array([x, y, f])
        z = np.array([0, 0, 1])
        t = self.get_t()
        # print(t)
        rot_flow = np.cross(z, np.cross(r, np.cross(w, r)))/f
        tran_flow = np.cross(z, np.cross(t, r))
        # print(rot_flow[0:2], tran_flow[0:2])
        return rot_flow[0:2], tran_flow[0:2]
    
    def cal_Z(self, x, y, normal):
        # print(normal, pts)
        norm = normal/np.linalg.norm(normal)
        # print(np.linalg.norm(norm))
        u_rot, u_tr = self.cal_flow(x, y)
        Z = (np.dot(u_tr, norm))/(np.linalg.norm(normal) - np.dot(u_rot, norm))
        print(Z)
        return Z
    
    def depth_img(self):
        cord, undistort_cord, new_n_flow = self.compute_new_normal()
        img = np.zeros((480, 640))
        for idx in range(len(cord)):
            Z = self.cal_Z(undistort_cord[idx][0], undistort_cord[idx][1], 
                           new_n_flow[idx])
            img[int(cord[idx][0]), int(cord[idx][1])] = Z
        plt.imshow(img)
        plt.show()




    
        
    
if __name__ == '__main__':
    example = get_t_depth(200)
    example.get_time_stamps()
    example.nearest_sample_index_rotation()
    example.nearest_sample_index_W
    # example.compute_new_normal()
    # example.cal_Z()
    example.depth_img()
