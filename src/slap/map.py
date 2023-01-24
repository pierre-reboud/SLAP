import numpy as np
from slap.view import View
from slap.point import Point

from typing import List, Any, Dict

class Map:

    def __init__(self, **kwargs):
        # Dict of all the frames in the 3d map
        self.cameras : np.ndarray = np.eye(4, dtype = np.float64)[None]#Dict[int, View] = {}
        # Dict of all the points in the 3d map
        self.points : np.ndarray = np.empty((0,4), np.float64)
        self.point_colors : np.ndarray = np.empty((0,3), np.float64)
        # Statistics
        self.statistics : Dict[str, Any] = {
            "n_points": 0,
            "n_view": 0,
        }

    def update(self, camera_pose : np.ndarray, spatial_points : np.ndarray, points_mask : np.ndarray, point_colors : np.ndarray):
        """_summary_

        Args:
            camera_pose (np.ndarray): world pose of the current view
            spatial_points (np.ndarray): array of size nx3 in the local view's
                frame of reference                  
            points_mask (np.ndarray): boolean masking array, where 0 
                represents points already included in the map and 1 points 
                that have yet to be included
        """        
        self._add_camera(camera_pose)
        self._add_points(spatial_points, point_colors)#[points_mask])


    def _add_camera(self, pose: np.ndarray) -> None:
        self.cameras = np.concatenate((self.cameras, (np.linalg.inv(pose)@self.cameras[-1])[None]), axis = 0)
        # import pdb
        # pdb.set_trace()
        # self._current_view : View = View(pose)
        # self.views.update({self._current_view.id, self._current_view})
        
    def _add_points(self, spatial_points: np.ndarray, point_colors : np.ndarray) -> None:
        self.points = np.concatenate((self.points, spatial_points), axis = 0)
        self.point_colors = np.concatenate((self.point_colors, point_colors), axis = 0)
        # new_point_ids = np.arange(len(self.points) , len(self.points) + len(spatial_points), dtype = np.uint32)
        # self._current_view.add_point_ids(new_point_ids)
        
        # for spatial_point in spatial_points:
        #     world_point : np.ndarray = np.linalg.inv(
        #         self.latest_pose)@np.concatenate(spatial_point,[1]) 
        #     point : Point = Point(world_point)
        #     self.points.update({point.id, point})
        #     self._current_view.add_point_id(point.id)

    @property
    def latest_pose(self) -> np.ndarray:
        return self._current_view.pose

    