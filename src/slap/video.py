import cv2
from typing import Dict, List, Generator
from slap.utils.utils import Configs
import numpy as np

class Video:
    def __init__(self, configs: Configs):
        self.path : str = configs.video_path
        self.configs = configs
        self.intrinsics : np.ndarray = self.build_intrinsics()
        self.distortion_coefs : np.ndarray = self.build_distortion_coefs()
        self.capture : cv2.VideoCapture = cv2.VideoCapture(self.path)
        self.frame_count : int = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_W : int = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_H : int = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stream = self.get_stream(self.path)
        self.orb = cv2.ORB_create(nfeatures = configs.n_features)
        # Buffer of two sequential frames
        if configs.grey:
            self.frames_buffer : np.ndarray = np.empty((2, self.frame_H, self.frame_W), dtype = np.uint8)
        else:
            self.frames_buffer : np.ndarray = np.empty((2, self.frame_H, self.frame_W, 3), dtype = np.uint8)
        self.keypoints_buffer : List = [None, None] #(2, 500, 32)
        self.descriptors_buffer : np.ndarray = np.empty((2, configs.n_features, configs.size_descriptor_buffer), dtype = np.uint8)        
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def get_stream(self, video_path: str) -> Generator[np.ndarray, str, None]:
        capture = cv2.VideoCapture(video_path)
        #buf : np.ndarray = np.empty((frame_count, frame_H, frame_W, 3), np.dtype('uint8'))
        frame_counter : int = 0
        frame_retrieved : bool = True
        while (frame_counter < self.frame_count and frame_retrieved):
            frame_retrieved, frame = capture.read()
            frame_counter += 1
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.undistort(frame, self.intrinsics, self.distortion_coefs)
            self.frames_buffer[0] = self.frames_buffer[1]
            self.frames_buffer[1] = frame
            yield frame
        capture.release()
        #cv2.namedWindow('frame 10')
        #cv2.imshow('frame 10', buf[9])
        #cv2.waitKey(0)
        return None

    def build_intrinsics(self) -> np.ndarray:
        intrinsics = np.eye(3)
        intrinsics[0,0] = self.configs.intrinsics.fx
        intrinsics[1,1] = self.configs.intrinsics.fy
        intrinsics[0,2] = self.configs.intrinsics.cx
        intrinsics[1,2] = self.configs.intrinsics.cy
        return intrinsics

    def build_distortion_coefs(self) -> np.ndarray:
        distortions = np.array([
            self.configs.intrinsics.k1,
            self.configs.intrinsics.k2,
            self.configs.intrinsics.p1,
            self.configs.intrinsics.p2,
            self.configs.intrinsics.k3
            ])
        return  distortions