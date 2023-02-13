import json 
import numpy as np
import cv2
from slap import __path__ 
from sys import exit
import os
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
from logging import debug, info
from scipy.spatial.transform import Rotation

@dataclass
class Configs:
    """Configs class to access config elements as attributes instead of a 
    subscriptable dictionary. Supports nested configurations.
    """
    def __init__(self, kwargs = None):
        """_summary_

        Args:
            kwargs (_type_, optional): _description_. Defaults to None.
        """
        root_path : List[str] = __path__[0].split("/")[:-2]
        configs_file_name = "freiburg_main_config.json" #"freiburg_main_config.json" #"video_main_config.json"
        data_name = "data/rgbd_dataset_freiburg1_xyz/" #"data/rgbd_dataset_freiburg1_xyz/" #"data" 
        kwargs_path : str = os.path.join("/", *root_path,"configs",configs_file_name)
        self.data_path : str = os.path.join("/", *root_path, data_name)     
        def local_setattr(down_self, down_kwargs):
            """_summary_

            Args:
                down_self (_type_): _description_
                down_kwargs (_type_): _description_
            """            
            for karg, varg in down_kwargs.items():
                if isinstance(varg, dict):
                    down_cfgs = DownConfigs()
                    print(down_self)
                    setattr(down_self, karg, down_cfgs)
                    local_setattr(down_cfgs, varg)
                else:
                    setattr(down_self, karg, varg)  
        if not kwargs:
            kwargs = Configs.get_args(kwargs_path)
        local_setattr(self, kwargs)
        debug(f"Configurations: {vars(self)}")
        self.camera_matrix : np.ndarray = np.array([
            [self.intrinsics.fx,0,self.intrinsics.cx],
            [0,self.intrinsics.fy,self.intrinsics.cy],
            [0,0,1]
            ])
        self.distortion_coefs : np.ndarray = np.array([
            self.intrinsics.k1,
            self.intrinsics.k2,
            self.intrinsics.p1,
            self.intrinsics.p2,
            self.intrinsics.k3
            ])
        self.video_path : str = os.path.join(self.data_path, self.video_name)
        # if self.video_name == "video.mp4":
        if self.video_name == "rgb":
            self._get_gt_cameras()

    @staticmethod
    def get_args(kwargs_path) -> Dict[str, Any]:
        """_summary_

        Returns:
            Dict[str, Any]: _description_
        """        
        with open(kwargs_path, "r") as j:
            kwargs : Dict[str, Any] = json.load(j)
            return kwargs
    
    def _get_gt_cameras(self) -> None:
        frame_names = set(sorted(os.listdir(self.video_path)))
        self._gt_cameras = np.zeros((len(frame_names),4,4), dtype = np.float64)
        j = 0
        with open(self.data_path + "/groundtruth.txt", "r") as f:
            for i, line in enumerate(f):
                if i < 3 or line[0] not in frame_names:
                    j+=1
                    continue
                line = line.split(" ")
                translations = line[1:4]
                #qx qy qz qw
                quaternions = line[4:]
                _rotation = Rotation.from_quat(quaternions)
                if i == 3:
                    self._gt_cameras[i-j][:3,:3] = _rotation.as_matrix()
                    self._gt_cameras[i-j][:3, 3] = np.array(translations, dtype = np.float64)
                _rotation = Rotation.from_quat(quaternions)
                self._gt_cameras[i-j][:3,:3] = self._gt_cameras[0][:3,:3].T@_rotation.as_matrix()
                self._gt_cameras[i-j][:3,3] = np.array(translations, dtype = np.float64) - self._gt_cameras[0][:3,3]
                self._gt_cameras[i-j][3,3] = 1.0

    def __repr__(self) -> str:
        return ""

@dataclass
class DownConfigs:
    def __init__(self):
        pass
    def __repr__(self) -> str:
        return ""

def signal_handler(sig, frame):
    info("Interrupt signal received! Exiting...")
    exit(0)


