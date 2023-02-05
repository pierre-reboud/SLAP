import numpy as np
import OpenGL.GL as gl
import pangolin
from time import sleep
from scipy.spatial.transform import Rotation
import signal
from slap.utils.utils import signal_handler 
from slap.utils.utils import Configs
import cv2
from logging import debug, warning
from typing import List, Any, Dict, Union
from threading import Thread
from multiprocessing import Process, Queue
from slap.map import Map
from sys import exit

class Visualizer:
    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """        
        self.configs : Configs = configs
        self.map_state : Union[None, Map] = None
        self.cams, self.points = [np.eye(4)], []
        self.queue : Queue = Queue()
        thread : Process = Process(target = self.run, args = [self.queue])
        thread.daemon = True
        thread.start()


    def draw_points(self, points : np.ndarray, colors : Union[np.ndarray, None] = None):
        """_summary_

        Args:
            points (np.ndarray): nx3 or nx4 array
            pose (np.ndarray): 4x4 array
        """
        # try:           
        #     assert (points.shape[1] == 3 or points.shape[1] == 4 and np.linalg.norm(points[:,3]-np.ones(points.shape[0])) < 10e-1)
        # except AssertionError:
        #     warning(f" Average of homogeneous index deviates significantly from 1. Deviation is: {np.linalg.norm(points[:,3]-np.ones(points.shape[0]))}")
        gl.glPointSize(4)
        gl.glColor3f(0, 1, 0)
        if points.shape[1] == 3:
            world_points = np.concatenate((points, np.ones(points.shape[0])[None,:].T), axis = 1)
        elif points.shape[1] == 4:
            world_points = points
        pangolin.DrawPoints(world_points[:,:3], colors)

    def draw_cams(self, poses: Union[np.ndarray, List[np.ndarray]]):
        """_summary_

        Args:
            poses (Union[np.ndarray, List[np.ndarray]]): _description_
            color (np.ndarray, optional): _description_. Defaults to None.
        """        
        # assert pose.shape == (4,4)
        # assert pose[3,3] == 1 
        # assert len(color) == 3 if color else True
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        poses = np.array(poses)
        pangolin.DrawCameras(poses)
    
    def draw(self, map: Map):
        self.queue.put(map)

    def run(self, queue : Queue):
        """_summary_

        Args:
            points (Union[np.ndarray, None], optional): np array of mx3 pose vectors. Defaults to None.
            cams (Union[np.ndarray,None], optional): np array of nx4x4 pose matrices. Defaults to None.
        """
        signal.signal(signal.SIGINT, signal_handler)
        self._init_pangolin()
        if self.configs.visualization._3d:
            while not pangolin.ShouldQuit():
                while not queue.empty():
                    self.map_state = queue.get()  
                #if not pangolin.ShouldQuit() and (not points is None or not cams is None) and self.configs.visualization_3d :
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glClearColor(1.0, 1.0, 1.0, 1.0)
                self.dcam.Activate(self.scam)
                if self.map_state is not None: 
                    if not len(self.map_state.points) == 0:
                        self.draw_points(points = self.map_state._fixed_points_3d, colors = self.map_state._fixed_point_colors)
                    if not len(self.map_state.cameras) == 0:
                        self.draw_cams(poses = self.map_state.cameras)
                pangolin.FinishFrame()
                
                
    
    def _dummy_run(self):
        """Dummy example, where 10000 randomly sampled 3d points are drawn within 
        a cube, as well as 5 cameras with varying pose.
        """        
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)
            
            # Render OpenGL Cube
            # pangolin.glDrawColouredCube()

            # Draw Point Cloud
            points = np.random.random((10000, 3)) * 10
            self.draw_points(points)

            pose= np.repeat(np.eye(4)[None,:],5,axis = 0)
            pose[:,0:3, 0:3] = np.array([Rotation.random().as_matrix() for _ in range(5)])
            debug(pose[:,0:3, 0:3].shape, pose.shape)
            pose[:,0:3,3] = np.array([np.random.rand(3)-0.5 for _ in range(5)])
            self.draw_cams(pose)

            pangolin.FinishFrame()

    def _draw_matched_frame(self, points_2d_older : np.ndarray, points_2d_newer : np.ndarray, frame : np.ndarray) -> None:
        """Draws the 2d frames with the corresponding point correspondences 

        Args:
            points_older (np.ndarray): _description_
            points_newer (np.ndarray): _description_
            frame (_type_): _description_
        """        
        if self.configs.visualization._2d:
            for pt1, pt2 in zip(points_2d_older, points_2d_newer):
                # _ = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags = 0)         
                cv2.line(frame, pt1, pt2, color = (0,255,0),thickness = 2)
            cv2.imshow("SLAM oder so kp hab nicht aufgepasst", frame)
            cv2.waitKey(int(not self.configs.debug_frame_by_frame))

    def _init_pangolin(self):
        pangolin.CreateWindowAndBind('Pangolin Test', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Define Projection and initial ModelView matrix, (last param is distance view)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 5000),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)


if __name__ == "__main__":
    sv = Visualizer()
    sv._dummy_run()

