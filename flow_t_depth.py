import numpy as np
from scipy.spatial.transform import Rotation as R
from event_warp import load_data
from event_warp import load_events
from utils import path


path = path()
flow = np.load(path + '/dataset_normal_flow.npz')
p, t, xy = load_events(path)
meta, depth, mask = load_data(path, format='evimo2v2')
# print(flow['normal_flow_pca_0000000001'].shape)
# print(flow['normal_flow_event_count_0000000001'].shape)
# print(flow['normal_flow_event_timestamp_0000000001'].shape)
# print(flow['normal_flow_gt_0000000001'].shape)
# print(flow['normal_flow_gt_mask_0000000001'].shape)

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
        self.left = frame

    def get_normal_flow(self):
        normal_flow_gt = []
        for file in flow.files:
            if (('gt' in file) and ('mask' not in file)):
                normal_flow_gt.append(file)
        # print(np.count_nonzero(flow[normal_flow_gt[self.left]]))
        return flow[normal_flow_gt[self.left]]
    
    def get_time_stamps(self):
        t_left = flow['t'][self.left]
        return t_left

    def nearest_sample_index_W(self):
        t_left= self.get_time_stamps()
        R_time = generate_W_t()
        idx = np.searchsorted(R_time, t_left)
        # print(idx)
        return idx
    
    def nearest_sample_index_rotation(self):
        t_left= self.get_time_stamps()
        R_time = generate_rotation_t()
        idx = np.searchsorted(R_time, t_left)
        # print(idx)
        return idx
    
    def get_t(self):
        idx = self.nearest_sample_index_rotation()
        # q_1 = meta['full_trajectory'][idx]['cam']['pos']['q']
        # R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],])
        # R_wc1 = R_wc1.as_matrix()
        T_1 = meta['full_trajectory'][idx-1]['cam']['pos']['t']
        T_2 = meta['full_trajectory'][idx+1]['cam']['pos']['t']
        time = meta['full_trajectory'][idx+1]['cam']['ts'] - meta['full_trajectory'][idx-1]['cam']['ts']
        # print(time)
        v_1 = np.array([T_1['x'], T_1['y'], T_1['z']])
        v_2 = np.array([T_2['x'], T_2['y'], T_2['z']])
        return (v_2 + v_1)/time
    
    def get_W(self):
        idx = self.nearest_sample_index_W()
        x = -meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['x']
        y = -meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['y']
        z = meta['imu']['/prophesee/left/imu'][idx]['angular_velocity']['z']
        W = [x, y, z]
        W = np.array(W)
        return W
    
    def cal_flow(self, x, y):
        f = meta['meta']['fy']
        # print(f)
        w = np.array(self.get_W())
        r = np.array([x, y, f])
        z = np.array([0, 0, 1])
        t = self.get_t()
        rot_flow = np.cross(z, np.cross(r, np.cross(w, r)))/f
        tran_flow = np.cross(z, np.cross(t, r))
        # print(rot_flow[0:2], tran_flow[0:2])
        return rot_flow[0:2], tran_flow[0:2]
    
    def cal_Z(self, x, y):
        normal_frame = self.get_normal_flow()
        print(normal_frame[x, y])
        print(normal_frame.shape)
        normal = normal_frame[x, y]
        norm = normal/np.linalg.norm(normal)
        # print(np.linalg.norm(norm))
        u_rot, u_tr = self.cal_flow(x, y)
        Z = (np.dot(u_tr, norm))/(np.linalg.norm(normal) - np.dot(u_rot, norm))
        print(Z)
        return Z



    
        
    
if __name__ == '__main__':
    example = get_t_depth(200)
    example.get_time_stamps()
    example.nearest_sample_index_rotation()
    example.nearest_sample_index_W()
    example.cal_Z(394, 381)
