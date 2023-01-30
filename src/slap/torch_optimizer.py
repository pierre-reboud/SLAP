# From README https://github.com/uoip/g2opy
import numpy as np
from slap.utils.utils import Configs
import torch
from torch import Tensor
from torch.optim import Adam
import g2o
from typing import Dict, Tuple, Any, Union, List

class Optimizer():
    def __init__(self, configs : Configs):
        self.configs : Configs = configs

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
        # Torchify parameters (3d points and cameras)
        cameras : Tensor = torch.tensor(cameras, requires_grad = True)
        list_of_points_2d : List[Tensor] = [torch.tensor(pts_2d) for pts_2d in points_2d]      
        points_3d : List[Tensor] = torch.cat([torch.tensor(pts_3d) for pts_3d in points_3d])
        points_3d = torch.tensor(points_3d, requires_grad = True)
        # Listify back the pts 3d tensor
        list_of_points_3d : List[Tensor]= list(torch.tensor_split(points_3d, _pts_3d_group_lengths))
        
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
        optimizer = Adam([points_3d, cameras], lr=1e-3)
        # Initialize loss function
        loss_func : torch.nn.MSELosss = torch.nn.MSELoss(reduction = "none")
        for i in range(self.configs.optimizer.iterations):
            optimizer.zero_grad()
            # Compute projection of 3D points
            projections : List[Tensor] = self.project(points_3d_optim, cameras)
            # Compute reprojection error
            pt_wise_loss : Tensor
            total_loss : float
            total_loss = self.reprojection_error(projections, points_2d_optim, loss_func)
            # Prune points
            #self.prune_points(pt_wise_loss, lengths_per_frame)
            # Compute gradients
            total_loss.backward()
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
            _intermediate_projections = _intermediate_projections[:, :3]@(torch.from_numpy(np.linalg.inv(self.configs.camera_matrix)).T)
            # nx2 / (2xn).T = nx2 / (nx2) = nx2 
            cam_projections = _intermediate_projections[:,:2] / _intermediate_projections[:,2].repeat(2,1).T
            
            all_projections.append(cam_projections)
        return all_projections
        

    def reprojection_error(self, projections : Tensor, points_2d : Tensor, loss_func : torch.nn.MSELoss) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            projections (Tensor): nx2 Tensor
            points_2d (Tensor): nx2 Tensor
            correspondences (Tensor): _description_
        """
        total_loss = 0.0
        for i, (projs, pts_2d) in enumerate(zip(projections, points_2d)):
            pt_wise_loss : Tensor = loss_func(input = projs, target = pts_2d).mean(dim=1)
            projective_loss : float = pt_wise_loss.mean(dim=0)
            # Implement reprojection error calculation
            total_loss += projective_loss
        return total_loss[None]
    
    def prune_points(self, pt_wise_los : Tensor, lengths_per_frame):
        pass

    def reformat_tensors(self, points_3d : Tensor, cameras: Tensor, lengths_per_frame):
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
        """Mother of all index fuckery!!!!
        Don't try to understand this one in detail if you want to stay sane.
        Basically, groups the 3d points (newly observed in current or previous 
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
        points_2d_to_optimize : List[Tensor] = []
        points_3d_to_optimize : List[Tensor] = []
        # Iterate from newest to oldest frame
        for i, (key, value)  in enumerate(reversed(correspondences.items())):
            # pts_3d, which are newly observed in frame i
            points_3d_to_optimize.append([])
            points_2d_to_optimize.append([])
            # Current frame index is key length - 1 - index
            curr_frame_ind = len(correspondences.keys()) - i - 1
            # Append newly seen pts to current frame's optimization pts
            points_3d_to_optimize[i].append(points_3d[curr_frame_ind])
            points_2d_to_optimize[i].append(points_2d[curr_frame_ind][value["corrs_to_prev"]==-1])
            
            # If there is a previous frame, also append all 3d pts that were seen before (but still in opt window)
            if i < len(correspondences.keys()) - 1:
                # Go to past index
                j = 0
                # While there are 2d_points in the old frame that correspond to non new points, and that are in current frame, go deeper in the past.  
                old_cors_prev = correspondences[key-j-1]["corrs_to_prev"]
                curr_cors_prev = correspondences[key-j]["corrs_to_prev"]
                while np.intersect1d(np.where(old_cors_prev != -1), curr_cors_prev[curr_cors_prev!=-1]).size != 0 or j == curr_frame_ind:
                    # Add current frame's 3d pts, which are older seen (first intersect term) and 
                    points_3d_to_optimize[i].append(points_3d[curr_frame_ind - j][np.intersect1d(np.where(old_cors_prev != -1), curr_cors_prev[curr_cors_prev!=-1])])
                    
                    # While there are old 3d pts seen in current frame, get their current frame indices 
                    # Go from old to current frame index
                    curr_cors_curr = correspondences[key-j]["corres_to_curr"]
                    _current_frame_2d_pts_indices = curr_cors_curr[np.intersect1d(np.where(old_cors_prev != -1), curr_cors_prev[curr_cors_prev!=-1])]
                    for k in range(0,j):
                        _current_frame_2d_pts_indices = correspondences[key-j+k]["corrs_to_curr"][_current_frame_2d_pts_indices]
                    points_2d_to_optimize[i].append(points_2d[_current_frame_2d_pts_indices])

                    j += 1
                    curr_cors_prev = np.intersect1d(np.where(old_cors_prev != -1), curr_cors_prev[curr_cors_prev!=-1]) 
                    old_cors_prev = correspondences[key-j]["corrs_to_prev"]
            import pdb
            pdb.set_trace()
            points_3d_to_optimize[i] = torch.cat(list(reversed(points_3d_to_optimize[i])), dim = 0) 
            points_2d_to_optimize[i] = torch.cat(list(reversed(points_2d_to_optimize[i])), dim = 0)    

        return points_2d_to_optimize, points_3d_to_optimize
    
