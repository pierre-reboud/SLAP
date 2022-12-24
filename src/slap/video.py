import cv2
from typing import Dict, List, Generator
from slap.utils.utils import Configs
import numpy as np

class Video:
    def __init__(self, configs: Configs):
        self.path : str = configs.video_path
        self.capture : cv2.VideoCapture = cv2.VideoCapture(self.path)
        self.frame_count : int = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_W : int = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_H : int = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stream = self.get_stream(self.path)
        self.orb = cv2.ORB_create(nfeatures = configs.n_features)
        # Buffer of two sequential frames
        self.frames_buffer : np.ndarray = np.empty((2, self.frame_H, self.frame_W), dtype = np.uint8)
        self.keypoints_buffer : List = [None, None] #(2, 500, 32)
        self.descriptors_buffer : np.ndarray = np.empty((2, configs.n_features, configs.size_descriptor_buffer), dtype = np.float32)        
        self.matcher = cv2.BFMatcher()

    def get_stream(self, video_path: str) -> Generator[np.ndarray, str, None]:
        capture = cv2.VideoCapture(video_path)
        #buf : np.ndarray = np.empty((frame_count, frame_H, frame_W, 3), np.dtype('uint8'))
        frame_counter : int = 0
        frame_retrieved : bool = True
        while (frame_counter < self.frame_count and frame_retrieved):
            frame_retrieved, frame = capture.read()
            frame_counter += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames_buffer[0] = self.frames_buffer[1]
            self.frames_buffer[1] = gray_frame
            #cv2.imshow("c'Est ma bite", gray_frame)
            #cv2.waitKey(1)
            yield gray_frame
        capture.release()
        #cv2.namedWindow('frame 10')
        #cv2.imshow('frame 10', buf[9])
        #
        #cv2.waitKey(0)
        return None