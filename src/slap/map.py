import numpy as np
from slap.view import View
from slap.point import Point
from slap.torch_optimizer import Optimizer
from slap.utils.utils import Configs

from typing import List, Any, Dict, Tuple
from logging import debug, info
from collections import OrderedDict

class Map:

    global_frame_index : int = 0
    
    def __init__(self, configs: Configs):
        self.configs : Configs = configs
        # nx4x4 Array of all the frames in the 3d map
        self.cameras : np.ndarray = np.empty((100,4,4), dtype = np.float64)
        self.cameras[0] : np.ndarray = np.eye(4, dtype = np.float64)[None]#Dict[int, View] = {}
        # nx4 Array of all the points in the 3d map
        self._fixed_points_3d : np.ndarray = np.empty((0,4), np.float64)
        self._optimizer_points_3d : List[np.ndarray] = []
        # Dictionary frame_id mapping to point locations and correspondings to previous id
        # Order is preserved for Python >=3.7 
        self._correspondences : Dict[int, Dict[str, np.ndarray]] = {}
        # nx3 Array of the point colors
        self._fixed_point_colors : np.ndarray = np.empty((0,3), np.float32)
        self._optimizer_point_colors : List[np.ndarray] = []
        # Statistics
        self._statistics : Dict[str, Any] = {
            "n_points": 0,
            "n_view": 0,
        }
        self.optimizer : Optimizer = Optimizer(self.configs)

    def update(self, camera_pose : np.ndarray, points_3d : np.ndarray,
        points_2d_older : np.ndarray, points_2d_newer : np.ndarray,
        correspondences_to_current : np.ndarray,
        correspondences_to_prev : np.ndarray, point_colors : np.ndarray):
        """_summary_

        Args:
            camera_pose (np.ndarray): world pose of the current view
            spatial_points (np.ndarray): array of size nx3 in the local view's
                frame of reference                  
            points_mask (np.ndarray): boolean masking array, where 0 
                represents points already included in the map and 1 points 
                that have yet to be included
        """
        # Update map frame index
        Map.global_frame_index += 1
        # Get the mask of the newly appearing points in current frame
        new_points_mask = correspondences_to_prev < 0
        # Mask the 2d points to get the ones corresponding to new 3d points
        point_2d_indices_of_new_3d_points = np.where(new_points_mask)
        
        self._add_camera(
            camera_pose = camera_pose
            )
        self._add_points_3d(
            points_3d = points_3d[new_points_mask],
            point_colors = point_colors[new_points_mask]
            )
        self._add_points_2d(
            points_2d_older = points_2d_older,
            points_2d_newer = points_2d_newer,
            new_points_mask = new_points_mask,
            correspondences_to_prev= correspondences_to_prev,
            correspondences_to_current= correspondences_to_current
        )
        info(f"New points: {new_points_mask.sum()}, \n Already mapped points {len(new_points_mask)-new_points_mask.sum()}, \n Total points: {self.points.shape[0]}")

    def optimize(self) -> None:
        """_summary_
        """      

        optimizeable_cameras = self.cameras[max(0,Map.global_frame_index - self.configs.optimizer.window_size): Map.global_frame_index]
        optimizeable_3d_points = self._optimizer_points_3d
        #points_2d_to_optimize, points_3d_to_optimize = self._get_optimization_points_subsets()
        points_2d_to_optimize = [self._correspondences[k]["pts_2d"] for k in self._correspondences.keys()]
        correspondences = self._correspondences
        optimized_3d_points : List[np.ndarray]
        optimized_cameras : np.ndarray 
        optimized_3d_points, optimized_cameras = self.optimizer.bundle_adjust(
            points_2d = points_2d_to_optimize,
            points_3d = optimizeable_3d_points,
            cameras = optimizeable_cameras,
            correspondences = correspondences
            )
        self._merge_with_map(optimized_3d_points, optimized_cameras)

    def _add_camera(self, camera_pose: np.ndarray) -> None:
        """_summary_

        Args:
            camera_pose (np.ndarray): _description_
        """
        if Map.global_frame_index == self.cameras.shape[0] - 1:
            _additional_space : np.ndarray =  np.empty(((Map.global_frame_index + 1)*2,4,4), dtype = np.float64)
            self.cameras = np.concatenate((self.cameras, _additional_space), axis = 0)       
        self.cameras[Map.global_frame_index] = np.linalg.inv(camera_pose)@(self.cameras[Map.global_frame_index - 1])
        
    def _add_points_3d(self, points_3d: np.ndarray, point_colors : np.ndarray) -> None:
        """_summary_

        Args:
            points_3d (np.ndarray): _description_
            point_colors (np.ndarray): _description_
        """        
        if len(self._optimizer_points_3d) == self.configs.optimizer.window_size:
            self._fixed_points_3d = np.concatenate((self._fixed_points_3d, self._optimizer_points_3d.pop(0)), axis = 0)
            self._fixed_point_colors = np.concatenate((self._fixed_point_colors, self._optimizer_point_colors.pop(0)), axis = 0)
        self._optimizer_points_3d.append(points_3d)
        self._optimizer_point_colors.append(point_colors)
        #self.point_colors = np.concatenate((self.point_colors, point_colors), axis = 0)
    
    def _add_points_2d(self, points_2d_older : np.ndarray, points_2d_newer: np.ndarray,
        correspondences_to_prev : np.ndarray, correspondences_to_current: np.ndarray,
        new_points_mask : np.ndarray) -> None:
        """_summary_

        Args:
            points_2d_older (np.ndarray): _description_
            points_2d_newer (np.ndarray): _description_
            correspondences_to_prev (np.ndarray): _description_
            correspondences_to_current (np.ndarray): _description_
        """        
        # if points_2d_older is not None:
        #     self._correspondences.update(
        #         {
        #             0: {
        #             "pts_2d" : points_2d_older,
        #             "new_pts_mask" : new_points_mask,
        #             "corrs_to_prev" : correspondences_to_prev,
        #             "corrs_to curr" : correspondences_to_current  
        #             }
        #         }
        #     )
        self._correspondences.update(
            {
                Map.global_frame_index: {
                    "pts_2d" : points_2d_newer,
                    "new_pts_mask" : new_points_mask,
                    "corrs_to_prev" : correspondences_to_prev,
                    "corrs_to curr" : correspondences_to_current  
                }
            }
        )
        if Map.global_frame_index == self.configs.optimizer.window_size:
            # + 1 because we compare an index with a length
            self._correspondences.pop(Map.global_frame_index + 1 - self.configs.optimizer.window_size)


    def _merge_with_map(self, optimized_3d_points : List[np.ndarray], optimized_cameras : np.ndarray) -> None:
        """_summary_

        Args:
            optimized_3d_points (List[np.ndarray]): _description_
            optimized_cameras (np.ndarray): _description_
        """
        # Merge cameras
        self.cameras[Map.global_frame_index-self.configs.optimizer.window_size : Map.global_frame_index] = optimized_cameras        
        # Merge 3d_points
        self._optimizer_points_3d = optimized_3d_points

    @property
    def points(self) -> np.ndarray:
        return self._fixed_points_3d




    