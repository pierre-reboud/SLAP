import json 
import numpy as np
import cv2
from slap import __path__ 
from sys import exit
import os
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
from logging import debug, info

root_path : List[str] = __path__[0].split("/")[:-2]
kwargs_path : str = os.path.join("/", *root_path,"configs","main_config.json")
data_path : str = os.path.join("/", *root_path,"data")

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
            kwargs = Configs.get_args()
        local_setattr(self, kwargs)
        debug(f"Configurations: {vars(self)}")
        self.video_path : str = os.path.join(data_path, self.video_name)
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


    @staticmethod
    def get_args() -> Dict[str, Any]:
        """_summary_

        Returns:
            Dict[str, Any]: _description_
        """        
        with open(kwargs_path, "r") as j:
            kwargs : Dict[str, Any] = json.load(j)
            return kwargs
    
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

def strip(corrs):
    return corrs[corrs!=-1]
        


