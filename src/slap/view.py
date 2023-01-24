import numpy as np
from typing import Dict, List, Any, Tuple

class View:

    count : int = 0

    def __init__(self, pose : np.ndarray):
        # 4x4 matrix of the homogeneous view-pose 
        self._cam_pose : np.ndarray = pose
        # List of the ids of the points present on the view's frame
        self._point_ids : np.ndarray = np.empty()#List[int] = []
        # View id
        self._id : int = View.count
        # Increment the view count
        View.count += 1

    def add_point_ids(self, ids : np.ndarray):
        self._point_ids = np.uin32(ids)
    
    def _get_start_and_end_indices(self) -> Tuple[int, int]:
        return (self._point_ids[0], self._point_ids[-1])
    
    @property
    def id(self) -> int:
        return self._id

    def pose(self) -> np.ndarray:
        return self._cam_pose

    def __hash__(self) -> int:
        return hash(self.id)
    