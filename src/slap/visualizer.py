import numpy as np
import OpenGL.GL as gl
import pangolin
from time import sleep
from scipy.spatial.transform import Rotation
from slap.utils.utils import Configs
import cv2

from typing import List, Any, Dict, Union

class Visualizer:
    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """        
        self.configs : Configs = configs

        pangolin.CreateWindowAndBind('Pangolin Test', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)
        self.cams, self.points = [np.eye(4)], []


    def draw_points(self, pts : np.ndarray):
        """_summary_

        Args:
            pts (np.ndarray): _description_
        """        
        assert pts.shape[1] == 3
        gl.glPointSize(2)
        gl.glColor3f(0, 1, 0)
        pangolin.DrawPoints(pts)

    def draw_cams(self, poses: Union[np.ndarray, List[np.ndarray]], color: np.ndarray = None):
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

    def run(self, points : Union[np.ndarray, None] = None, cams : Union[np.ndarray,None] = None):
        """_summary_

        Args:
            points (Union[np.ndarray, None], optional): np array of mx3 pose vectors. Defaults to None.
            cams (Union[np.ndarray,None], optional): np array of nx4x4 pose matrices. Defaults to None.
        """        
        # while not pangolin.ShouldQuit():
        if not pangolin.ShouldQuit() and (not points is None or not cams is None) and self.configs.visualization_3d :
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)
            
            if not points is None and points.shape == (4,4):
                self.points.append(points)
                if self.configs.vis.size_pts_buffer < len(self.points):
                    self.points.pop(0)
                points = self.points
            if not cams is None and cams.shape == (4,4):
                self.cams.append(np.linalg.inv(cams)@self.cams[-1])
                if self.configs.vis.size_cam_buffer < len(self.cams):
                    self.cams.pop(0)
                cams = self.cams

            if not points is None:
                self.draw_points(points = points)
            if not cams is None:
                self.draw_cams(poses = cams)

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
            pangolin.glDrawColouredCube()

            # Draw Point Cloud
            points = np.random.random((10000, 3)) * 10
            self.draw_points(points)

            pose= np.repeat(np.eye(4)[None,:],5,axis = 0)
            pose[:,0:3, 0:3] = np.array([Rotation.random().as_matrix() for _ in range(5)])
            print(pose[:,0:3, 0:3].shape, pose.shape)
            pose[:,0:3,3] = np.array([np.random.rand(3)-0.5 for _ in range(5)])
            self.draw_cams(pose)

            pangolin.FinishFrame()

    def _draw_matched_frame(self, points_frame_1 : np.ndarray, points_frame_2 : np.ndarray, frame : np.ndarray) -> None:
        """Draws the 2d frames with the corresponding point correspondences 

        Args:
            points_frame_1 (np.ndarray): _description_
            points_frame_2 (np.ndarray): _description_
            frame (_type_): _description_
        """        
        #points_frame_1 = [np.uint16(self.video.keypoints_buffer[0][match[0].queryIdx].pt) for match in matches]
        #points_frame_2 = [np.uint16(self.video.keypoints_buffer[1][match[0].trainIdx].pt) for match in matches]
        if self.configs.visualization_2d:
            for pt1, pt2 in zip(points_frame_1, points_frame_2):
                # _ = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags = 0)         
                cv2.line(frame, pt1, pt2, color = (0,255,0),thickness = 2)
            cv2.imshow("SLAM oder so kp hab nicht aufgepasst", frame)
            cv2.waitKey(int(not self.configs.debug_frame_by_frame))


if __name__ == "__main__":
    sv = Visualizer()
    sv._dummy_run()

