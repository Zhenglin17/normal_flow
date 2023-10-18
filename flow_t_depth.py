import numpy as np

flow = np.load('/home/zhenglin/Documents/left_camera/sfm/eval/scene_03_00_000000/dataset_flow.npz')
print(len(flow['t']))
for i in range(100):
    print(flow['t'][i+1] - flow['t'][i])