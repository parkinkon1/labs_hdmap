import torch
from torch.utils.data import DataLoader

from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

import torch

from torch.utils.data import Dataset

import os
import pickle

from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from rasterization_q10.input_representation.static_layers import StaticLayerRasterizer
from rasterization_q10.input_representation.agents import AgentBoxesWithFadedHistory
from rasterization_q10 import PredictHelper

from rasterization_q10.helper import convert_global_coords_to_local
import matplotlib.pyplot as plt

from pkyutils import NusCustomParser



root='/datasets/nuscene/v1.0-mini'
version='v1.0-mini'
load_dir='../nus_dataset'


layer_names = ['drivable_area', 'road_segment', 'road_block',
                'lane', 'ped_crossing', 'walkway', 'stop_line',
                'carpark_area', 'road_divider', 'lane_divider']

colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
            (255, 255, 255), (255, 255, 255), (255, 255, 255), ]

DATAROOT = root
version = version
sampling_time = 3
agent_time = 0  # zero for static mask, non-zero for overlap

dataset = NusCustomParser(
            root=DATAROOT,
            version=version,
            sampling_time=sampling_time,
            agent_time=agent_time,
            layer_names=layer_names,
            colors=colors,
            resolution=0.1,
            meters_ahead=32,
            meters_behind=32,
            meters_left=32,
            meters_right=32)

print("num_samples: {}".format(len(dataset)))



idx = 0
map_masks, map_img, agent_mask, xy_local, virtual_mask, virtual_xy_local, idx = dataset[idx]
agent_past, agent_future, agent_translation = xy_local
fake_past, fake_future, fake_translation = virtual_xy_local














