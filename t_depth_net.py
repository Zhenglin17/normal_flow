import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

class Fetch_normal(Dataset):
    def __init__(self, img_path):
        self.image = read_image(img_path)
        self.num_channels, self.h, self.w = self.image.shape

    def __len__(self):
        ### TODO: 1 line of code for returning the number of pixels
        return self.h*self.w

    def __getitem__(self, idx):
        ### TODO: 2-3 lines of code for x, y, and pixel values
        x = idx % self.w
        y = idx // self.h
        intensity = self.image[:,y,x]

        return {"x": x, "y": y, "intensity": intensity}
    
class FFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, coord):
        out = self.fc1(coord)
        out = self.relu(out)
        out = self.fc2(out)
        return out