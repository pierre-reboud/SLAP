import numpy as np
import OpenGL.GL as gl
import pangolin
from time import sleep
from scipy.spatial.transform import Rotation

class Spatial_Visualizer:
    def __init__(self, points : np.ndarray = None, cams : np.ndarray = None):
        
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


    def draw_points(self, pts : np.ndarray):
        assert pts.shape[1] == 3
        gl.glPointSize(2)
        gl.glColor3f(0, 1, 0)
        pangolin.DrawPoints(pts)

    def draw_cams(self, poses: np.ndarray, color: np.ndarray = None):
        # assert pose.shape == (4,4)
        # assert pose[3,3] == 1 
        # assert len(color) == 3 if color else True
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawCameras(poses)

    def run(self, points : np.ndarray = None, cams : np.ndarray = None):
        """
        cams = np array of nx4x4 pose matrices
        points = np array of mx3 pose vectors
        """
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)
            
            self.draw_points(points)
            self.draw_cam(cams)

            pangolin.FinishFrame()
    
    def _dummy_run(self):
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
            self.draw_cam(pose)

            pangolin.FinishFrame()

if __name__ == "__main__":
    sv = Spatial_Visualizer()
    sv._dummy_run()

