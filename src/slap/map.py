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
        self.views : List[View] = [None, None, None, None, None]  # 5 Views, a good map point should be able to track in at least 4 views
        self.statistics : Dict[str, Any] = {
            "n_points": 0,
            "n_view": 0,
        }
        self.matches_times : np.ndarray = np.empty((0, 0), np.uint16)

    def update(self, camera_pose : np.ndarray, spatial_points : np.ndarray, matches: List[List[Any]], point_colors : np.ndarray):
        """_summary_

        Args:
            camera_pose (np.ndarray): world pose of the current view
            spatial_points (np.ndarray): array of size nx3 in the local view's
                frame of reference                  
            points_mask (np.ndarray): boolean masking array, where 0 
                represents points already included in the map and 1 points 
                that have yet to be included
        """         
        # track the points 
        if self.views[-3]:
            points_ids : np.ndarray = self.views[-1]._point_ids
            old_points_ids = self.views[-2]._point_ids
            for idx, match in enumerate(matches):
                if old_points_ids[0, match[0].trainIdx]:
                    points_ids[0, match[0].queryIdx] = old_points_ids[0, match[0].trainIdx]
                    self.matches_times[old_points_ids[0, match[0].trainIdx]] += 1
                else: 
                    old_points_ids[0, match[0].trainIdx] = self.statistics["n_points"]
                    points_ids[0, match[0].queryIdx] = self.statistics["n_points"]
                    self.statistics["n_points"] += 1
                    # add new point into mappoints
                    self.points = np.concatenate((self.points, [spatial_points[idx, :]]), axis = 0)
                    # add matched times for the points
                    self.matches_times = np.append(self.matches_times, 1)
        else:
            # add all the points into map points
            self._add_points(spatial_points, point_colors)#[points_mask])
            self.statistics["n_points"] = spatial_points.shape[0]
            # also add all the points to view 0 and view 1
            points_ids0 : np.ndarray = self.views[-2]._point_ids
            points_ids1 : np.ndarray = self.views[-1]._point_ids
            for i, match in enumerate(matches):
                points_ids0[0, match[0].trainIdx] = i
                points_ids1[0, match[0].queryIdx] = i
            # set all points matched for 2 times
            self.matches_times = np.ones(spatial_points.shape[0], np.uint16)
            
        self._add_camera(camera_pose)

        # culling point with number of matches under 3
        if self.views[0]:
            self.points[self.matches_times[np.max(self.views[0]._point_ids):np.max(self.views[1]._point_ids)], :] = np.array([0,0,0,1])
        elif self.views[1]:
            boolean_array = self.matches_times[:np.max(self.views[1]._point_ids) + 1] < 3
            full_true = np.full(self.points.shape[0] - np.max(self.views[1]._point_ids) - 1, True)
            new_ba = np.concatenate((boolean_array, full_true))
            self.points[new_ba, :] = np.array([0,0,0,1])
        # self._add_points(spatial_points, point_colors)#[points_mask])


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

    