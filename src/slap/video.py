import cv2
from typing import Dict, List, Generator, Tuple
from slap.utils.utils import Configs
import numpy as np

class Video:
    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """        
        self.path : str = configs.video_path
        self.configs = configs
        self.intrinsics : np.ndarray = self.configs.camera_matrix
        self.distortion_coefs : np.ndarray = self.configs.distortion_coefs
        self.capture : cv2.VideoCapture = cv2.VideoCapture(self.path)
        self.frame_count : int = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_W : int = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_H : int = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.stream = self.get_stream(self.path)
        self.orb = cv2.ORB_create(nfeatures = configs.n_features)
        self.matcher = cv2.BFMatcher()
        # Buffer of two sequential frames
        _buffer_size = (3, self.frame_H, self.frame_W) if configs.grey else (3, self.frame_H, self.frame_W,3) # (2, self.frame_H, self.frame_W) if configs.grey else (2, self.frame_H, self.frame_W,3)
        self.frames_buffer : np.ndarray = np.empty(_buffer_size, dtype = np.uint8)
        self.descriptors_buffer : np.ndarray = np.empty((3, configs.n_features, configs.size_descriptor_buffer), dtype = np.float32) # np.empty((2, configs.n_features, configs.size_descriptor_buffer), dtype = np.float32)
        self.keypoints_buffer : List = [None, None, None] #[None, None] #(2, 500, 32)      
        

    def get_stream(self, video_path: str) -> Generator[Tuple[np.ndarray, int, int], str, None]:
        """Yields tuple of the frame, the buffer index of the newer (first) and of the older (second) frame.

        Args:
            video_path (str): _description_

        Returns:
            _type_: _description_

        Yields:
            Generator[Tuple[np.ndarray, int, int], str, None]: _description_
        """               
        capture = cv2.VideoCapture(video_path)
        frame_counter : int = 0
        frame_retrieved : bool = True
        frame : np.ndarray
        while (frame_counter < self.frame_count and frame_retrieved):
            frame_retrieved, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.configs.grey else frame
            frame = cv2.undistort(frame, self.intrinsics, self.distortion_coefs)
            self.frames_buffer[frame_counter%3] = frame
            yield frame, frame_counter%3, (frame_counter-1)%3
            frame_counter += 1
        capture.release()
        #cv2.namedWindow('frame 10')
        #cv2.imshow('frame 10', buf[9])
        #cv2.waitKey(0)
        return None
