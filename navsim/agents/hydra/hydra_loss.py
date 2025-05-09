from typing import Dict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex


def hydra_loss(targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig):
    
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """

    #without imitation score-----------------------------------------------------
    trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    loss = (
        config.trajectory_weight * trajectory_loss
        + config.bev_semantic_weight * bev_semantic_loss
    )
    #without imitation score-----------------------------------------------------
#
    #with imitation score------------------------------------------------------
    """

    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    #print("Pred_trajectory shape: ", pred_trajectories.shape) # (1000, 8, 3) -> 1000 vocab_size
    #print("Target trajectory shape: ", target_trajectories.shape) # (64, 8, 3) -> 64 batch_size
    #print("Imitation_scores shape: ", imitation_scores.shape)
    #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_trajectories = predictions["trajectory"].to(device) #(60, 24, 3): 60 trajectory (fixed), 24 pos, 3 (x,y,theta)
    target_trajectories = targets["trajectory"].to(device) #(64, 24, 3): 64 batch_size, 24 pos, 3 (x,y,theta)
    imitation_scores = predictions["imitation_score"].to(device)

    total_imitation_loss = 0

    for i in range(target_trajectories.size(0)):  # batch size = 64
        target_trajectory = target_trajectories[i]  # (8, 3)
        imitation_score = imitation_scores[i]  # (1000,1)

        # Expand target trajectory to match predicted trajectories
        expanded_target_trajectory = target_trajectory.expand(pred_trajectories.size(0), -1, -1)  # (1000, 8, 3)

        pred_trajectories_reshaped = pred_trajectories.reshape(pred_trajectories.shape[0], -1)
        expanded_target_trajectory_reshaped = expanded_target_trajectory.reshape(expanded_target_trajectory.shape[0], -1)
        # Compute trajectory-wise distance
        distance = torch.norm(pred_trajectories_reshaped - expanded_target_trajectory_reshaped, p = 2, dim=-1)


        # Compute softmax-based imitation target
        imitation_targets = torch.softmax(-distance ** 2, dim=-1)

        # Compute softmax over imitation scores
        imitation_score = imitation_score.squeeze(1)  # Ensure shape is (1000,)
        imitation_score = F.softmax(imitation_score, dim=-1)  # (1000,)

        # Compute weighted negative log-likelihood loss
        imitation_loss = -torch.sum(imitation_targets * torch.log(imitation_score + 1e-8))  # Prevent log(0)

        total_imitation_loss += imitation_loss

    #total_imitation_loss = total_imitation_loss / target_trajectories.size(0)  
    loss = (
        #config.trajectory_weight * trajectory_loss
        total_imitation_loss
    )
    """
    #with imitation score--------------------------------------------------------------------


    return loss
