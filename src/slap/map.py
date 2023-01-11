import numpy as np
from typing import List

class Point:
  def __init__(self, map):
    self.frames = []
    self.idxs = []
    self.id = map.add_point(self)

  def add_observation(self, frame, idx):
    if frame.pts[idx] is not None:
        # TODO: duplicated match needs to be solved
        return
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

class Frame:
    def __init__(self, map, kps : np.ndarray, des : np.ndarray) -> None:
       self.kps : np.ndarray = kps  # keypoints
       self.des : np.ndarray = des  # descriptors
       self.pts : List[Point] = [None]*len(self.kps)   # relative points
       self.id : int = map.add_frame(self)

class Map:
    def __init__(self) -> None:
        self.frames : List[Frame] = []
        self.map_points : List[Point] = []
        self.num_frames = 0
        self.num_points = 0

    def add_point(self, point) -> int:
        self.map_points.append(point)
        ret = self.num_points
        self.num_points += 1
        return ret

    def add_frame(self, frame) -> int:
        self.frames.append(frame)
        ret = self.num_frames
        self.num_frames += 1
        return ret
    