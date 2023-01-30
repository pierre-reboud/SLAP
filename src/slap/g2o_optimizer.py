# From README https://github.com/uoip/g2opy
import numpy as np
from slap.utils.utils import Configs
import g2o

class Optimizer(g2o.SparseOptimizer):
    def __init__(self, configs : Configs):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        self.configs : Configs = configs

    def optimize(self, max_iterations : int = 10):
        """_summary_

        Args:
            max_iterations (int, optional): _description_. Defaults to 10.
        """        
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id : int, pose : np.ndarray, fixed : bool = False) -> None:
        """_summary_

        Args:
            pose_id (int): _description_
            pose (np.ndarray): _description_
            cam (np.ndarray): _description_
            fixed (bool, optional): _description_. Defaults to False.
        """        
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(self.configs.intrinsics.fx, self.configs.intrinsics.fy, self.configs.intrinsics.cx, self.configs.intrinsics.cy, 0)#cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id : int, point : np.ndarray, fixed=False, marginalized=True) -> None:
        """_summary_

        Args:
            point_id (int): _description_
            point (np.ndarray): _description_
            fixed (bool, optional): _description_. Defaults to False.
            marginalized (bool, optional): _description_. Defaults to True.
        """        
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id : int, pose_id : int, 
            measurement : np.ndarray,
            information : np.ndarray = np.identity(2),
            robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))) -> None:   # 95% CI
        """_summary_

        Args:
            point_id (int): _description_
            pose_id (int): _description_
            measurement (np.ndarray): _description_
            information (np.ndarray, optional): _description_. Defaults to np.identity(2).
            robust_kernel (_type_, optional): _description_. Defaults to g2o.RobustKernelHuber(np.sqrt(5.991)).
        """        

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id : int):
        """_summary_

        Args:
            pose_id (int): _description_

        Returns:
            _type_: _description_
        """        
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id : int):
        """_summary_

        Args:
            point_id (int): _description_

        Returns:
            _type_: _description_
        """        
        return self.vertex(point_id * 2 + 1).estimate()