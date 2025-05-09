from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_callback import TransfuserCallback
from navsim.agents.hydra.hydra_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.agents.hydra.hydra_loss import hydra_loss
from navsim.agents.hydra.HydraModel import HydraModel
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.agents.hydra.PlanningVocabulary import PlanningVocabulary
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class HydraAgent(AbstractAgent):
    """Agent for HydraMDP"""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes HydraMDP agent.
        :param config: global config of HydraMDP agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._hydra_model = HydraModel(self._trajectory_sampling, config, device)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]

        #print("AAAAAAAAAAAAAAAAa")
        #for k, v in state_dict.items():
        #    if "lidar" in k and "latent" in k:
        #        print(k)
        #print(state_dict)

        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: HydraMDP only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._hydra_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return hydra_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""

        return torch.optim.Adam(self._hydra_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return None
        #return [TransfuserCallback(self._config)]
    
    """
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        #Computes the ego vehicle trajectory.
        #:param current_input: Dataclass with agent inputs.
        #:return: Trajectory representing the predicted ego's position in future
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = {k: v.to(device) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            weights = {"imitation_weight": 1}

            predictions = self.forward(features)
            imitation_scores = predictions["imitation_score"]

            imitation_weight = weights["imitation_weight"]
            weights = torch.tensor(imitation_weight, dtype=imitation_scores.dtype, device=imitation_scores.device).unsqueeze(0)

            inference_scores = - (
                #weights * torch.log(imitation_scores)
                torch.log(imitation_scores)
            )


            inference_scores[inference_scores.isnan()] = 1e9 

            
            _min_value, min_index = inference_scores.min(dim=1)
            print("Minimum inference_scores: ", min_index, ", ", _min_value)
            print(inference_scores)
            poses = predictions["trajectory"][min_index].squeeze(0).cpu().numpy()

            poses = poses.squeeze(0)
            

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)
        """
