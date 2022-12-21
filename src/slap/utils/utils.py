import json 
import numpy as np
import cv2
from slap import __path__ 

import sys
import os
from typing import List, Dict, Any, Generator
from dataclasses import dataclass

root_path : List[str] = __path__[0].split("/")[:-2]
kwargs_path : str = os.path.join("/", *root_path,"configs","main_config.json")
data_path : str = os.path.join("/", *root_path,"data")

@dataclass
class Config:

    def __init__(self, kwargs = None):
        if not kwargs:
            kwargs = Config.get_args()
        for karg, varg in kwargs.items():
            setattr(self, karg, varg)
        print(vars(self))
        self.video_path = os.path.join(data_path, self.video_name)

    @staticmethod
    def get_args() -> Dict[str, Any]:
        with open(kwargs_path, "r") as j:
            kwargs : Dict[str, Any] = json.load(j)
            return kwargs


