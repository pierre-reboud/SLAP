# From README https://github.com/uoip/g2opy
import numpy as np
from slap.utils.utils import Configs, strip
import torch
from torch import Tensor
from torch.optim import Adam
import g2o
from typing import Dict, Tuple, Any, Union, List
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Queue, Process
from time import sleep
from datetime import datetime

class Optimizer():
    def __init__(self, configs : Configs):
        self.configs : Configs = configs
        # self.queue : Queue = Queue()
        # sw_process = Process(target = self._summary_write, args = [self.queue])
        # sw_process.daemon = True
        # sw_process.start()     

    def bundle_adjust(self, points_2d : List[Tensor], points_3d : List[np.ndarray],
        cameras : np.ndarray, correspondences : Dict[int, Dict[str, np.ndarray]]
        ) -> Tuple[List[np.ndarray], np.ndarray]:
        """_summary_

        Args:
            points_3d (List[np.ndarray]): _description_
            cameras (np.ndarray): _description_
            correspondences (Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]): _description_

        Returns:
            _type_: _description_
        """
        # Get the pts_3d corresponding list (also gets updated, as underlying 
        # data is same as points_3d tensor)
        _pts_3d_group_lengths = [len(pts_3d) for pts_3d in points_3d]
        for i, e in enumerate(_pts_3d_group_lengths):
            _pts_3d_group_lengths[i] = e if i == 0 else e + _pts_3d_group_lengths[i-1]  
        # Torchify parameters (3d points and cameras)
        cameras : Tensor = torch.tensor(cameras, requires_grad = True)
        list_of_points_2d : List[Tensor] = [torch.tensor(pts_2d) for pts_2d in points_2d]      
        points_3d : Tensor = torch.cat([torch.tensor(pts_3d) for pts_3d in points_3d])
        points_3d = torch.tensor(points_3d, requires_grad = True)
        # _dcopy_pts_3d = points_3d.detach().clone()
        # Listify back the pts 3d tensor (:-1)
        list_of_points_3d : List[Tensor]= list(torch.tensor_split(points_3d, _pts_3d_group_lengths[:-1]))
        # Group to know which 3d points to project in which frame. The different 
        # name to points_xd is done, so that we can keep the points_xd tensor,
        # which also gets updated due to gradient flowing though all Tensor 
        # operations

        points_2d_optim, points_3d_optim = self._group_pts_per_frame(
            points_2d = list_of_points_2d,
            points_3d = list_of_points_3d,
            correspondences = correspondences
            )

        # Initialize optimizer and 3D points tensor
        optimizer = Adam([points_3d, cameras], lr=self.configs.optimizer.lr)
        # Initialize loss function
        loss_func : torch.nn.MSELosss = torch.nn.MSELoss(reduction = "none")
        for i in range(self.configs.optimizer.iterations):
            optimizer.zero_grad()
            # Compute projection of 3D points
            projections : List[Tensor] = self.project(points_3d_optim, cameras)
            # Compute reprojection error
            pt_wise_loss : Tensor
            total_loss : float
            total_loss = self.reprojection_error(projections, points_2d_optim, loss_func, adam_iter = i)
            # Prune points
            #self.prune_points(pt_wise_loss, lengths_per_frame)
            # Compute gradients
            total_loss.backward(retain_graph = True)
            # Update parameters
            optimizer.step()
        points_3d : List[np.ndarray] = [pts_3d.detach().numpy() for pts_3d in list_of_points_3d]
        cameras : np.ndarray = cameras.detach().numpy()

        return points_3d, cameras

    def project(self, points_3d : List[Tensor], cameras : Tensor) -> List[Tensor]:
        """_summary_

        Args:
            points_3d (List[Tensor]): List of len [optimizer_window_size] containing the 3d points 
            cameras (Tensor): _description_
        """
        # Go from group of 3d points observed first to last
        all_projections : List[Tensor] = []
        for i, (cam, pts_3d) in enumerate(zip(cameras, points_3d)):
            # nx4@4x4 = 4x4
            _intermediate_projections = pts_3d@(cam.T)
            # nx3@3x3
            #_intermediate_projections = _intermediate_projections[:, :3]@(torch.from_numpy(np.linalg.inv(self.configs.camera_matrix)).T)
            # nx2 / (2xn).T = nx2 / (nx2) = nx2 
            cam_projections = _intermediate_projections[:,:2] / _intermediate_projections[:,2].repeat(2,1).T
            
            all_projections.append(cam_projections)
        return all_projections
        

    def reprojection_error(self, projections : Tensor, points_2d : Tensor, loss_func : torch.nn.MSELoss, adam_iter : int) -> Tuple[Tensor, Tensor]:
        """Computes the MSE reprojection error

        Args:
            projections (Tensor): nx2 Tensor
            points_2d (Tensor): nx2 Tensor
            correspondences (Tensor): _description_
        """
        total_loss = 0.0
        pt_wise_losses = []
        for i, (projs, pts_2d) in enumerate(zip(projections, points_2d)):
            # if adam_iter == 10:
            # breakpoint()
            pt_wise_loss : Tensor = loss_func(input = projs, target = pts_2d).mean(dim=1)
            projective_loss : float = pt_wise_loss.mean(dim=0)
            # Implement reprojection error calculation
            pt_wise_losses.append(pt_wise_loss)
            total_loss += projective_loss
        pt_wise_losses = torch.cat(pt_wise_losses, dim=0).clone().detach()
        # self.queue.put({
        #     "loss": total_loss.clone().detach()/len(points_2d),
        #     "max_pixel_error": np.linalg.norm(self.configs.camera_matrix[:,:2]@(projs[torch.argmax(pt_wise_loss,)]-pts_2d[torch.argmax(pt_wise_loss,)]).clone().detach().numpy()),
        #     "errors" : self.configs.intrinsics.fx*pt_wise_loss.clone().detach(),
        #     "iter": adam_iter
        #     })
        return total_loss[None]
    
    def prune_points(self, pt_wise_los : Tensor, lengths_per_frame):
        pass

    def reformat_tensors(self, points_3d : Tensor, cameras: Tensor, lengths_per_frame):
        """_summary_

        Args:
            points_3d (Tensor): _description_
            cameras (Tensor): _description_
            lengths_per_frame (_type_): _description_

        Returns:
            _type_: _description_
        """        
        cameras = cameras.numpy()
        points_3d : List[np.ndarray] = []
        start_index : int = 0
        for i, block_length in enumerate(lengths_per_frame):
            _block : np.ndarray = points_3d[start_index:start_index + block_length]
            points_3d.append(_block)
            start_index += block_length
        return points_3d, cameras
    

        
    def _group_pts_per_frame(self, points_2d : List[Tensor], points_3d : List[Tensor],
        correspondences : Dict[int, Dict[str, np.ndarray]]
        ) -> Tuple[List[Tensor], List[Tensor]]:
        """Groups the 3d points (newly observed in current or previous 
        frames) that match the current 2d pts. As not all 2d pts in the current
        frame have a corresponding 3d pt in the optimization window (3d point
        might be older than oldest optimization window frame), we also need to
        build the subset of 2d pts on the basis of which we want to perform the
        optimization.
        Also, O(n^3), so super slow. A better way might exist??

        Args:
            points_2d (List[Tensor]): _description_
            points_3d (List[Tensor]): _description_
            correspondences (List[int, Dict[str, np.ndarray]]): _description_
            lengths_2d_pts_per_frame (Dict[int, int]): _description_
            lengths_3d_pts_per_frame (Dict[int, int]): _description_
            frame_index (int): _description_

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: _description_
        """     
        key : int
        value : Dict[str, np.ndarray]
        # Point groups for optimization
        points_2d_to_optimize : List[Tensor] = [[] for _ in range(len(correspondences.keys()))]
        points_3d_to_optimize : List[Tensor] = [[] for _ in range(len(correspondences.keys()))]
        # Iterate from oldest to newest frame and fill up the list of point 
        # groups over which the optimization is performed
        for i, (key, value)  in enumerate(correspondences.items()):
            # Append newly seen pts to current frame's optimization pts
            try:
                assert i < len(points_3d) and len(points_3d[i]) == (value["corrs_to_prev"]==-1).sum()
            except AssertionError:
                breakpoint()
            points_3d_to_optimize[i].append(points_3d[i])
            points_2d_to_optimize[i].append(points_2d[i][value["corrs_to_prev"]==-1])
            # Go to future frames
            j = 1
            while key + j in correspondences.keys(): 
                _indices_pts_3d_to_select = self._get_correspondences_to_past_indices(correspondences, start = key, end = key + j)
                _indices_pts_2d_to_select = self._get_past_indices_to_current_indices(correspondences, start_indices = _indices_pts_3d_to_select, start = key, end = key + j)
                assert len(_indices_pts_2d_to_select) == len(_indices_pts_3d_to_select)
                if _indices_pts_3d_to_select.size == 0:
                    points_3d_to_optimize[i + j].append(points_3d[i][_indices_pts_3d_to_select])
                    points_2d_to_optimize[i + j].append(points_2d[i + j][_indices_pts_2d_to_select])
                j += 1
            points_3d_to_optimize[i] = torch.cat(list(points_3d_to_optimize[i]), dim = 0) 
            points_2d_to_optimize[i] = torch.cat(list(points_2d_to_optimize[i]), dim = 0)
        for group2d, group3d in zip(points_2d_to_optimize, points_3d_to_optimize):
            try:
                assert len(group2d) == len(group3d)
            except AssertionError:
                breakpoint()
        return points_2d_to_optimize, points_3d_to_optimize
  
    def _get_correspondences_to_past_indices(self, correspondences : Dict[int, Dict[str, np.ndarray]], start : int, end : int):
        """Computes the indices in the current frame that are present along the 
        entire optimization window.

        Args:
            correspondences (Dict[int, Dict[str, np.ndarray]]): _description_
            start (int): _description_
            end (int): _description_

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        if start == end - 1:
            indices_where_new = np.where(correspondences[start]["corrs_to_prev"]==-1)[0]
            indices_from_curr = correspondences[end]["corrs_to_prev"][correspondences[end]["corrs_to_prev"]!=-1]
            indices_intersected = indices_from_curr[np.isin(indices_from_curr, indices_where_new)]
            return indices_intersected
        elif start < end -1:
            indices_from_end = correspondences[end]["corrs_to_prev"][correspondences[end]["corrs_to_prev"]!=-1]
            indices_where_new = np.where(correspondences[start]["corrs_to_prev"]==-1)[0]
            while start < end - 1:
                indices_where_not_new = np.where(correspondences[end-1]["corrs_to_prev"] != -1)[0]
                indices_intersect = indices_from_end[np.isin(indices_from_end,indices_where_not_new)]
                indices_from_end = correspondences[end-1]["corrs_to_prev"][indices_intersect]
                end -= 1 
            indices = indices_from_end[np.isin(indices_from_end, indices_where_new)]
            return indices
        else:
            raise Exception(f"Start index must be at least 1 smaller than end index, currently: start:{start} end:{end}")
    
    def _get_past_indices_to_current_indices(self, correspondences : Dict[int, Dict[str, np.ndarray]], start_indices, start: int, end : int):
        """Computes the indices in the current frame that are present along the 
        entire optimization window.

        Args:
            correspondences (Dict[int, Dict[str, np.ndarray]]): _description_
            start (int): _description_
            end (int): _description_
        """
        #print("Wesh       " , start, end, start_indices)
        if start < end:
            while start < end:
                start_indices = correspondences[start + 1]["corrs_to_curr"][start_indices]
                start += 1 
            return start_indices
        else:
            raise Exception(f"Start index must be at least 1 smaller than end index, currently: start:{start} end:{end}")

    def _summary_write(self, queue : Queue):
        """Writes the content of the multiprocessing queue to the tensorboard
        SummaryWriter object.
        Args:
            queue (Queue): _description_
        """
        folder = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") 
        self.sw = SummaryWriter(log_dir = self.configs.tensorboard_logdir_path + "/" + folder)
        while True:
            while not queue.empty():
                things = queue.get()
                loss, max_pixel_dist, reproj_errors, iter = things.values()
                self.sw.add_scalar("Loss_per_frame", loss, iter)
                self.sw.add_scalar("Max_pixel_dist", max_pixel_dist, iter)
                self.sw.add_histogram("Reprojection_errors", reproj_errors, iter)
            sleep(1)