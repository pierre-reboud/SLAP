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
from threading import Thread

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
    
    def run(self) -> NotImplemented:
        """_summary_

        Returns:
            NotImplemented: _description_
        """        
        # Get frame and index from video stream
        for index, frame in enumerate(self.video.stream):
            # Tuple of cv2 keypoint objects , descriptorts = nd array of nx32
            keypoints, descriptors = self.video.orb.detectAndCompute(frame, None)       
            #print("Shape: ", len(keypoints), descriptors.shape) 
            self.video.descriptors_buffer[0] = self.video.descriptors_buffer[1] 
            self.video.descriptors_buffer[1] = descriptors
            # Pops the first list elements
            self.video.keypoints_buffer.pop(0)
            self.video.keypoints_buffer.append(keypoints)
            if index!=0:
                # color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                color_frame = frame
                matches = self.get_matches()
                self.process_matches(matches, color_frame)

    def process_matches(self, matches : List[List[cv2.DMatch]], frame) -> None:
        """_summary_

        Args:
            matches (List[List[cv2.DMatch]]): _description_
            frame (_type_): _description_
        """                
        points_frame_1 : np.ndarray = np.array([np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches])
        points_frame_2 : np.ndarray =  np.array([np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches])

        # Cam pose is 4x4 pose
        cam_pose : np.ndarray = self._get_cam_pose(points_frame_1 = points_frame_1, points_frame_2 = points_frame_2)
        points_pose : np.ndarray = self._get_points_pose(points_frame_1, points_frame_2)

        # 2d visualization
        self.visualizer._draw_matched_frame(points_frame_1, points_frame_2, frame)
        # 3d visualization
        self.visualizer.run(points = None, cams = cam_pose)
        
        
        debug(f"R(euler) {Rotation.from_matrix(cam_pose[:3,:3]).as_euler('xyz', degrees = True)} \t t {cam_pose[:3,3]}")


    def get_matches(self) -> List[List[cv2.DMatch]]:
        """_summary_

        Returns:
            List[List[cv2.DMatch]]: _description_
        """        
        candidate_matches = self.video.matcher.knnMatch(self.video.descriptors_buffer[0], self.video.descriptors_buffer[1], k = 2)
        matches = self.apply_lowe_ratio(candidate_matches)
        return matches
    
    def apply_lowe_ratio(self, candidate_matches : List[List[cv2.DMatch]]) -> List[List[cv2.DMatch]]:
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
        return matches 

    def _get_cam_pose(self, points_frame_1 : np.ndarray, points_frame_2 : np.ndarray) -> np.ndarray:
        """Estimates the cam pose matrix in the coordinates of the previous frame

        Args:
            points1 (np.ndarray): _description_
            points2 (np.ndarray): _description_

        Returns:
            np.ndarray: 4x4 pose array in the coordinates of the previous pose
        """
        #print((cv2.findFundamentalMat.__doc__))
        
        F, _mask = cv2.findFundamentalMat(points_frame_1, points_frame_2, method = cv2.FM_RANSAC)
        E = self.configs.camera_matrix@F@self.configs.camera_matrix.T
        R1, R2,t = cv2.decomposeEssentialMat(E)
        t = t[:,0]
        # Hacky ambiguity resolution
        euler_angles_1 = Rotation.from_matrix(R1).as_euler('xyz', degrees = True)
        euler_angles_2 = Rotation.from_matrix(R2).as_euler('xyz', degrees = True)
        R = R1 if max(euler_angles_1) < max(euler_angles_2) else R2
        # Frame time interval
        T = 1/29.97
        if not hasattr(self, "prev_t"):
            self.prev_t = np.zeros(3)
        v = (self.prev_t - t) / T
        # If euler integration of speed is closer to negative vector
        t = t if np.linalg.norm(self.prev_t + v*T - t) < np.linalg.norm(self.prev_t + v*T - (-t)) else -t 
        self.prev_t = t
        # import pdb
        # pdb.set_trace()
        # R,t = self._essential_to_Rt(E)
        
        # E, _ = cv2.findEssentialMat(
        #     points1 = points_frame_1,
        #     points2 = points_frame_2,
        #     cameraMatrix = self.configs.camera_matrix,
        #     method = cv2.RANSAC
        #     )
        # _, R, t, _ = cv2.recoverPose(
        #     E = E,
        #     points1 = points_frame_1,
        #     points2 = points_frame_2, 
        #     cameraMatrix = self.configs.camera_matrix
        #     )
        cam_pose : np.ndarray = self._get_cam_pose_from_Rt(R, t)
        return cam_pose

    def _fundamental_to_Rt(self, F : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """From https://github.com/geohot/twitchslam

        Args:
            F (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """                              
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        U,d,Vt = np.linalg.svd(F, full_matrices = True)
        if np.linalg.det(U) < 0:
            U *= -1.0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]

        # TODO: Resolve ambiguities in better ways. This is wrong.
        if t[2] < 0:
            t *= -1
        return R, t

    def _essential_to_Rt(self, R : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """From https://ia601408.us.archive.org/view_archive.php?archive=/7/items/DIKU-3DCV2/DIKU-3DCV2.zip&file=DIKU-3DCV2%2FHandouts%2FLecture16.pdf slide 

        Args:
            F (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """                              
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
        U,d,Vt = np.linalg.svd(F, full_matrices = True)
        if np.linalg.det(U) < 0:
            U *= -1.0
        if np.linalg.det(Vt) < 0:
            Vt *= -1.0
        R = np.dot(np.dot(U, W), Vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(U, W.T), Vt)
        t = U[:, 2]

        # TODO: Resolve ambiguities in better ways. This is wrong.
        if t[2] < 0:
            t *= -1
        return R, t

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

    def _get_points_pose(self, points1, points2):
        """_summary_

        Args:
            points1 (_type_): _description_
            points2 (_type_): _description_
            R (_type_): _description_
            t (_type_): _description_
            E (_type_): _description_

        Returns:
            _type_: _description_
        """        
        # Use factorization method slide 75
        # http://vision.stanford.edu/teaching/cs231a_autumn1112/lecture/lecture10_multi_view_cs231a.pdf
        D = np.int16(np.concatenate((points1, points2), axis = 1).T)#np.empty((4,len(points1)))
        # import pdb
        # pdb.set_trace()
        D = (D.T-np.average(D,axis = 1)).T
        uu,ss,vh = np.linalg.svd(D, full_matrices = True)
        debug(f"u {uu.shape}, s {ss.shape}, vh {vh.shape}")
        # M is the projection matrix of the 3d points to 2d frame coordinates
        M = uu[:,:3]
        # S is the nx3 matrix of points in 3d space 
        S = (np.diag(ss[:-1])@vh[:3]).T
        # svd = TruncatedSVD(n_components = 3)
        # P = svd.fit_transform(D.T)
        return S

    def _append_to_map(self, points : np.ndarray, cam_pose : np.ndarray):
        view : View = View(pose = cam_pose)
        self.map.add_view(view)
          
    


