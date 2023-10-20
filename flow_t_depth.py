import numpy as np
from scipy.spatial.transform import Rotation as R
from event_warp import load_data
from event_warp import load_events


path = '/home/zhenglin/Documents/left_camera/sfm/eval/scene_03_00_000000'
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
    for index in range(len(meta['imu']['/prophesee/left/imu'])):
        time_list.append(meta['full_trajectory'][index]['cam']['ts'])
    time_list = np.array(time_list)
    return time_list

def generate_W_t():
    time_list = []
    for index in range(len(meta['full_trajectory'])):
        time_list.append(meta['imu']['/prophesee/left/imu'][index]['ts'])
    time_list = np.array(time_list)
    return time_list

class get_t_depth():
    def __init__(self, frame_1, frame_2):
        self.left = frame_1
        self.right = frame_2

    def get_normal_flow(self):
        normal_flow_gt = []
        for file in flow.files:
            if (('gt' in file) and ('mask' not in file)):
                normal_flow_gt.append(file)
        # print(len(normal_flow_gt))
        return normal_flow_gt[self.left], normal_flow_gt[self.right]
    
    def get_time_stamps(self):
        t_left = flow['t'][self.left]
        t_right = flow['t'][self.right]
        return t_left, t_right

    def nearest_sample_index_W(self):
        t_left, t_right = self.get_time_stamps()
        R_time = generate_W_t()
        idx = np.searchsorted(R_time, [t_left, t_right])
        # print(idx)
        return idx
    
    def nearest_sample_index_rotation(self):
        t_left, t_right = self.get_time_stamps()
        R_time = generate_rotation_t()
        idx = np.searchsorted(R_time, [t_left, t_right])
        # print(idx)
        return idx
    
    def get_rotation(self):
        idx = self.nearest_sample_index_rotation()
        q_1 = self.meta['full_trajectory'][idx[0]]['cam']['pos']['q']
        q_2 = self.meta['full_trajectory'][idx[1]]['cam']['pos']['q']
        R_wc1 = R.from_quat([q_1['x'], q_1['y'], q_1['z'], q_1['w'],])
        R_wc2 = R.from_quat([q_2['x'], q_2['y'], q_2['z'], q_2['w'],])
        R_wc1 = R_wc1.as_matrix()
        R_wc2 = R_wc2.as_matrix()
        R_c1c2 = R_wc1.T @ R_wc2
        return R_c1c2
    
    def get_W(self):
        idx = self.nearest_sample_index_W()
        x = -meta['imu']['/prophesee/left/imu'][idx[0]]['angular_velocity']['x']
        y = -meta['imu']['/prophesee/left/imu'][idx[0]]['angular_velocity']['y']
        z = meta['imu']['/prophesee/left/imu'][idx[0]]['angular_velocity']['z']
        W = [x, y, z]
        W = np.array(W)
        return W
    
    def cal_flow(self, x, y):
        f = meta['meta']['fy']
        w = np.array(self.get_W())
        r = np.array([x, y, f])
        z = np.array[0, 0, 1]
        rot_flow = np.cross(z, np.cross(r, np.cross(w, r)))/f
        
    
if __name__ == '__main__':
    example = get_t_depth(100, 200)
    example.get_normal_flow()
    example.get_time_stamps()
    example.rot_flow(101, 200)
