#!/home/pierre/envs/general/bin/python

import cv2
from slap.video import Video
from slap.map import Map
from slap.point import Point
from slap.view import View
from slap.utils.utils import Configs
from typing import Dict, List, Any, Union, Tuple
from logging import info, debug, warning
import logging
import numpy as np
import itertools
from sys import exit
from slap.visualizer import Visualizer

from scipy.spatial.transform import Rotation
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.DEBUG)

class Slam:
    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """        
        self.configs : Configs = configs
        self.video : Video = Video(configs)
        self.visualizer : Visualizer = Visualizer(configs)
        self.map : Map = Map()
        self.matches_buffer : List[List[cv2.DMatch]] = [] 
    
    def run(self) -> NotImplemented:
        """_summary_

        Returns:
            NotImplemented: _description_
        """        
        # Get frame and index from video stream
        frame: np.ndarray
        newest: int
        for index, (frame, newer, older) in enumerate(self.video.stream):
            # Tuple of cv2 keypoint objects , descriptorts = nd array of nx32
            keypoints, descriptors = self.video.orb.detectAndCompute(frame, None)       
            #print("Shape: ", len(keypoints), descriptors.shape) 
            
            # Add to queue
            self.video.descriptors_buffer[newer] = descriptors
            # self.video.keypoints_buffer.pop(newer)
            # self.video.keypoints_buffer.insert(newer, keypoints)
            self.video.keypoints_buffer.pop(0)
            self.video.keypoints_buffer.insert(2, keypoints)

            # Only match when two frames are available 
            if index!=0:
                self.get_matches(newer, older)
                self.process_matches(frame, newer, older, index)

    def process_matches(self, frame: np.ndarray, newer : int, older: int, index : int) -> None:
        """_summary_

        Args:
            matches (List[List[cv2.DMatch]]): _description_
            frame (np.ndarray): _description_
            newer (int): Index of the newer element in the buffer (0 or 1)
            older (int): Index of the older element in the buffer (0 or 1)
            index (int): Index of the current frame
        """                
        # points_older : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[older][match[0].trainIdx].pt) for match in matches])
        # points_newer : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[newer][match[0].queryIdx].pt) for match in matches])
        points_older : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in self.matches_buffer[-1]])
        points_newer : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[2][match[0].queryIdx].pt) for match in self.matches_buffer[-1]]) 

        normalized_points_older = (np.linalg.inv(self.configs.camera_matrix)@np.concatenate((points_older.T,np.ones(len(points_older))[None]), axis = 0))[:2].T
        normalized_points_newer = (np.linalg.inv(self.configs.camera_matrix)@np.concatenate((points_newer.T,np.ones(len(points_older))[None]), axis = 0))[:2].T

        point_colors = self.video.frames_buffer[newer,points_newer[:,1], points_newer[:,0]] / 255.0 
        # Cam pose is 4x4 pose (takes point in first, transforms into second coord sys)
        camera_pose : np.ndarray = self._get_cam_pose(
            points_older = points_older,
            points_newer = points_newer
            )
        points : np.ndarray = self._get_points_pose(
            points_older = normalized_points_older,
            points_newer = normalized_points_newer,
            new_cam_pose = camera_pose
            )

        # Map operations
        self.mapify(
            camera_pose = camera_pose,
            points = points,
            point_colors = point_colors,
            )
        # 2d visualization
        self.visualizer._draw_matched_frame(
            points_older = points_older,
            points_newer = points_newer,
            frame = frame
            )
        # 3d visualization
        self.visualizer.draw(map = self.map)
        debug(f"R(euler) {Rotation.from_matrix(camera_pose[:3,:3]).as_euler('xyz', degrees = True)} \t t {camera_pose[:3,3]}")


    def get_matches(self, newer : int, older : int) -> None:
        """_summary_

        Args:
            newer (int): Index of the newer element in the buffer (0 or 1)
            older (int): Index of the older element in the buffer (0 or 1)

        Returns:
            List[List[cv2.DMatch]]: _description_
        """        
        candidate_matches = self.video.matcher.knnMatch(
            queryDescriptors = self.video.descriptors_buffer[newer],
            trainDescriptors = self.video.descriptors_buffer[older], k = 2
            )
        self._apply_lowe_ratio(candidate_matches)
        if len(self.matches_buffer) > 2:
            self.matches_buffer.pop(0)
    
    def _apply_lowe_ratio(self, candidate_matches : List[List[cv2.DMatch]]) -> None:
        """_summary_

        Args:
            candidate_matches (_type_): _description_

        Returns:
            List[List[cv2.DMatch]]: _description_
        """        
        matches : List[List[cv2.DMatch]]= []
        for better, worse in candidate_matches:
            if better.distance < self.configs.lowe_ratio * worse.distance:
                matches.append([better])
        self.matches_buffer.append(matches)

    def _get_cam_pose(self, points_older : np.ndarray, points_newer : np.ndarray) -> np.ndarray:
        """Estimates the cam pose matrix in the coordinates of the previous frame
        (takes point in first, transforms into second coord sys)

        Args:
            points_older (np.ndarray): _description_
            points_newer (np.ndarray): _description_

        Returns:
            np.ndarray: 4x4 pose array in the coordinates of the previous pose
        """
        # Hacky ambiguity resolution
        if self.configs.ambiguity_resolution == "cv2":
            E, _ = cv2.findEssentialMat(
                points2 = points_newer,
                points1 = points_older,
                cameraMatrix = self.configs.camera_matrix,
                method = cv2.RANSAC
                )
            # From first camera to second (takes point in first, transforms into second coord sys)
            _, R, t, _ = cv2.recoverPose(
                E = E,
                points2 = points_newer,
                points1 = points_older, 
                cameraMatrix = self.configs.camera_matrix
                )
            t = t[:,0]
        # Cv2 ambiguity resolution
        elif self.configs.ambiguity_resolution == "hacky":
            F, _mask = cv2.findFundamentalMat(
                points1 = points_newer,
                points2 = points_older,
                method = cv2.FM_RANSAC)
            E = self.configs.camera_matrix.T@F@self.configs.camera_matrix
            R1, R2,t = cv2.decomposeEssentialMat(E)
            t = t[:,0]
            euler_angles_1 = Rotation.from_matrix(R1).as_euler('xyz', degrees = True)
            euler_angles_2 = Rotation.from_matrix(R2).as_euler('xyz', degrees = True)
            R = R1 if max(euler_angles_1) < max(euler_angles_2) else R2
            # Frame time interval
            T = 1/29.97
            if not hasattr(self, "prev_t"):
                self._prev_t = np.zeros(3)
            v = (self._prev_t - t) / T
            # If euler integration of speed is closer to negative vector
            t = t if np.linalg.norm(self._prev_t + v*T - t) < np.linalg.norm(self._prev_t + v*T - (-t)) else -t 
            self._prev_t = t
        cam_pose : np.ndarray = self._get_cam_pose_from_Rt(R, t)
        return cam_pose

    def _get_cam_pose_from_Rt(self, R, t) -> np.ndarray:
        """_summary_

        Args:
            R (_type_): _description_
            t (_type_): _description_

        Returns:
            np.ndarray: 4x4 pose array in the coordinates of the previous pose
        """              
        cam_pose : np.ndarray = np.eye(4)
        cam_pose[:3, :3] = R
        cam_pose[:3,3] = t
        return cam_pose

    def _get_points_pose(self, points_older : np.ndarray, points_newer : np.ndarray, new_cam_pose : np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            points_older (_type_): _description_
            points_newer (_type_): _description_
            R (_type_): _description_
            t (_type_): _description_
            E (_type_): _description_

        Returns:
            _type_: _description_
        """
        old_cam_pose = self.map.cameras[-1]

        # http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/lecture10_multi_view_cs231a.pdf
        # Use factorization method slide 75
        if self.configs.points_extraction == "factorization":
            # D = [x,y,x',y']^Txn
            D = np.int16(np.concatenate((points_older, points_newer), axis = 1).T)
            # 4-array of [x,y,x',y'] centroid
            d_avg = np.average(D,axis = 1)
            # D centered in centroid and transposed -> nx4
            D = (D.T-d_avg).T 
            uu,ss,vh = np.linalg.svd(D, full_matrices = True)
            # M is the projection matrix of the 3d points to 2d frame coordinates
            M = uu[:,:3]
            # S is the nx3 matrix of points in 3d space 
            S = (np.diag(ss[:-1])@vh[:3]).T
            points_4d = np.concatenate((S,np.ones(len(points_older))[None].T), axis = 1)

        # As in page 330 of Book Multiple View Geometry
        elif self.configs.points_extraction == "linear_triangulation":
            points_4d : np.ndarray = np.zeros((len(points_older),4))
            A = np.zeros((4 , 4 * len(points_newer)))
            A[0 , :] = np.outer(points_older[: , 0], old_cam_pose[2]).flatten()[None] - np.repeat(old_cam_pose[0][None], len(points_older), axis = 0).flatten()
            A[1 , :] = np.outer(points_older[: , 1], old_cam_pose[2]).flatten()[None] - np.repeat(old_cam_pose[1][None], len(points_older), axis = 0).flatten()
            A[2 , :] = np.outer(points_newer[: , 0], new_cam_pose[2]).flatten()[None] - np.repeat(new_cam_pose[0][None], len(points_older), axis = 0).flatten()
            A[3 , :] = np.outer(points_newer[: , 1], new_cam_pose[2]).flatten()[None] - np.repeat(new_cam_pose[1][None], len(points_older), axis = 0).flatten()
            for i, (pn, po) in enumerate(zip(points_newer, points_older)):
                # Get nullspace vector (x, where Ax = 0, multiple possible vectors)
                # last vh row is vector corresponding to (close to) null singular value
                u, s, vh = np.linalg.svd(A[:,i*4:(i+1)*4], full_matrices = True)
                points_4d[i] = vh[3] / vh[3,3]
            # u, s, vh = np.linalg.svd(A, full_matrices = True)
            # points_4d : np.ndarray = np.reshape(vh[-1], (-1, 4))
            # points_4d = points_4d / (np.repeat(points_4d[:,3][None], 4, axis = 0)).T
            # points_4d = points_4d@(old_cam_pose.T) # Because (A@P^T)^T = (P@A^T) 
        return points_4d

    def mapify(self, points : np.ndarray, camera_pose : np.ndarray, point_colors : np.ndarray):
        # points_older : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[0][match[0].trainIdx].pt) for match in matches])
        # points_newer : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[1][match[0].queryIdx].pt) for match in matches]) 
        self.map.update(
            camera_pose = camera_pose,
            spatial_points = points,
            points_mask = None,
            point_colors = point_colors
            )
            
          
    


