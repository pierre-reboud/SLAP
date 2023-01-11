import numpy as np
from slap.view import View
from slap.point import Point

from typing import List, Any, Dict

class Map:

    def __init__(self, **kwargs):
        # Dict of all the frames in the 3d map
        self.views : Dict[int, View] = {}
        # Dict of all the points in the 3d map
        self.points : Dict[int, Point] = {} 

    def update(self, camera_pose : np.ndarray, spatial_points : np.ndarray, points_mask : np.ndarray):
        """_summary_

        Args:
            camera_pose (np.ndarray): world pose of the current view
            spatial_points (np.ndarray): array of size nx3 in the local view's
                frame of reference                  
            points_mask (np.ndarray): boolean masking array, where 0 
                represents points already included in the map and 1 points 
                that have yet to be included
        """        
        self._add_view(camera_pose)
        self._add_points(spatial_points[points_mask])


    def _add_view(self, pose: np.ndarray):
        self._current_view : View = View(pose)
        self.views.update({self._current_view.id, self._current_view})
        
    def _add_points(self, spatial_points: np.ndarray):
        for spatial_point in spatial_points:
            world_point : np.ndarray = np.linalg.inv(
                self.latest_pose)@np.concatenate(spatial_point,[1]) 
            point : Point = Point(world_point)
            self.points.update({point.id, point})
            self._current_view.add_point_id(point.id)

    @property
    def latest_pose(self):
        return self._current_view.pose

    