#!/home/pierre/envs/general/bin/python

import cv2
from slap.video import Video
from slap.utils.utils import Configs
from typing import Dict, List, Any, Union
from logging import info, debug, warning
import logging
import numpy as np
import itertools
from sys import exit
from slap.visualizer import Spatial_Visualizer
from slap.map import Map, Point, Frame
from threading import Thread

from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.DEBUG)

class Slam:
    def __init__(self, configs: Configs):
        self.configs : Configs = configs
        self.video : Video = Video(configs)
        self.spatial_visualizer : Spatial_Visualizer = Spatial_Visualizer(configs)
        self.map : Map = Map()
    
    def run(self) -> NotImplemented:
        pass
    
    def _test_run(self) -> None:
        for index, frame in enumerate(self.video.stream):
            keypoints, descriptors = self.video.orb.detectAndCompute(frame, None)
            frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags = 0) 
            #print("Shape: ", len(keypoints), descriptors.shape) 
            self.video.descriptors_buffer[0] = self.video.descriptors_buffer[1] 
            self.video.descriptors_buffer[1] = descriptors
            # Pops the first list elements
            self.video.keypoints_buffer.pop(0)
            self.video.keypoints_buffer.append(keypoints)
            current_frame = Frame(self.map, keypoints, descriptors)
            if index!=0:
                # color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                previous_frame = self.map.frames[-2]
                color_frame = frame
                matches = self.get_matches()
                # add matched points to map
                for match in matches:
                    if previous_frame.pts[match[0].queryIdx] is None:
                        pt = Point(self.map)
                        pt.add_observation(previous_frame, match[0].queryIdx)
                        pt.add_observation(current_frame, match[0].trainIdx)
                    elif previous_frame.pts[match[0].queryIdx] is not None and current_frame.pts[match[0].trainIdx] is None:
                        previous_frame.pts[match[0].queryIdx].add_observation(current_frame, match[0].trainIdx)
                        current_frame.pts[match[0].trainIdx] = previous_frame.pts[match[0].queryIdx]
                self.process_matches(matches, color_frame)
                

    def process_matches(self, matches, frame) -> None:
        points_frame_1 = np.array([np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches])
        points_frame_2 = np.array([np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches])
        self.draw_matched_frame(points_frame_1, points_frame_2, frame)
        #print((cv2.findFundamentalMat.__doc__))
        #F = cv2.findFundamentalMat(points_frame_1, points_frame_2, method = cv2.FM_RANSAC)
        E, _ = cv2.findEssentialMat(
            points1 = points_frame_1,
            points2 = points_frame_2,
            cameraMatrix = self.configs.camera_matrix,
            method = cv2.RANSAC
            )
        _, R, t, _ = cv2.recoverPose(
            E = E,
            points1 = points_frame_1,
            points2 = points_frame_2, 
            cameraMatrix = self.configs.camera_matrix
            )
        #R1, R2,t = cv2.decomposeEssentialMat(E)
        if self.configs.visualization_3d:
            cam_pose, points = self._get_cam_and_points(points_frame_1, points_frame_2, R, t, E)
            self.spatial_visualizer.run(points = None, cams = cam_pose)
        debug(f"R(euler) {Rotation.from_matrix(R).as_euler('xyz', degrees = True)} \t t {t[:,0]}")


    def draw_matched_frame(self, points_frame_1, points_frame_2, frame) -> None:
        #points_frame_1 = [np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches]
        #points_frame_2 = [np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches]
        for pt1, pt2 in zip(points_frame_1, points_frame_2):
            cv2.line(frame, pt1, pt2, color = (0,255,0),thickness = 2)
        for pt in filter(None, self.map.frames[-1].pts):
            if len(pt.frames) > 2:
                for f1, idx1, f2, idx2 in zip(pt.frames, pt.idxs, pt.frames[1:], pt.idxs[1:]):
                    cv2.line(frame, np.uint16(f1.kps[idx1].pt), np.uint16(f2.kps[idx2].pt), color = (255,0,0),thickness = 1)
        if self.configs.visualization_2d:
            cv2.imshow("SLAM oder so kp hab nicht aufgepasst", frame)
            cv2.waitKey(int(not self.configs.debug_frame_by_frame))

    def get_matches(self) -> List[List[cv2.DMatch]]:
        candidate_matches = self.video.matcher.knnMatch(self.video.descriptors_buffer[0], self.video.descriptors_buffer[1], k = 2)
        matches = self.apply_lowe_ratio(candidate_matches)
        return matches
    
    def apply_lowe_ratio(self, candidate_matches) -> List[List[cv2.DMatch]]:
        matches = []
        for better, worse in candidate_matches:
            if better.distance < self.configs.lowe_ratio * worse.distance:
                matches.append([better])
        return matches 

    def _test_view(self) -> None:
        for frame in itertools.islice(self.video.stream, 100):
            cv2.imshow("2d-Frame", frame)
            cv2.waitKey(int(not self.configs.debug_frame_by_frame))

    def _get_cam_and_points(self, points1, points2, R, t, E):
        cam_pose : np.ndarray = np.eye(4)
        point_cloud = np.empty((points1.shape[0],3))
        cam_pose[:3, :3] = R
        cam_pose[:3,3] = t[:,0]
        # for pt1, pt2 in zip(points1, points2):
            # import pdb
            # pdb.set_trace()
        return cam_pose, None

            


