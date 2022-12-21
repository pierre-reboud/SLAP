#!/home/pierre/envs/general/bin/python

import cv2
from slap.video import Video
from slap.utils.utils import Config
from typing import Dict, List
import numpy as np
import itertools
from sys import exit

class Slam:
    def __init__(self, config: Config):
        self.configs : Config = config
        self.video : Video = Video(config)
    
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
            if index!=0:
                color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                matches = self.get_matches()
                self.process_matches(matches, color_frame)
                cv2.imshow("SLAM oder so kp hab nicht aufgepasst", color_frame)
                cv2.waitKey(1)

    def process_matches(self, matches, frame) -> None:
        points_frame_1 = np.array([np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches])
        points_frame_2 = np.array([np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches])
        self.draw_matched_frame(points_frame_1, points_frame_2, frame)
        #print((cv2.findFundamentalMat.__doc__))
        F = cv2.findFundamentalMat(points_frame_1, points_frame_2, method = cv2.FM_RANSAC)
        E, _ = cv2.findEssentialMat(points_frame_1, points_frame_2, focal = 1, pp = np.zeros((2)), method = cv2.RANSAC)#, param1 = 3, param2 = 0.99)
        R1, R2,t = cv2.decomposeEssentialMat(E)
        print(R1, "\t", R2, "\t", t.T)


    def draw_matched_frame(self, points_frame_1, points_frame_2, frame) -> None:
        #points_frame_1 = [np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches]
        #points_frame_2 = [np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches]
        for pt1, pt2 in zip(points_frame_1, points_frame_2):
            cv2.line(frame, pt1, pt2, color = (0,255,0),thickness = 2)

    def get_matches(self) -> List[List[cv2.DMatch]]:
        candidate_matches = self.video.matcher.knnMatch(self.video.descriptors_buffer[0], self.video.descriptors_buffer[1], k = 2)
        matches = self.apply_lowe_ratio(candidate_matches)
        return matches
    
    def apply_lowe_ratio(self, candidate_matches) -> List[List[cv2.DMatch]]:
        matches = []
        for better, worse in candidate_matches:
            #if better.distance < self.kwargs["slam"]["lowe_ratio"] * worse:
            if better.distance < 0.5 * worse.distance:
                matches.append([better])
        return matches 

    def _test_view(self) -> None:
        for frame in itertools.islice(self.video.stream, 100):
            cv2.imshow("2d-Frame", frame)
            cv2.waitKey(0)


