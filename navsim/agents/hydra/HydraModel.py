from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from navsim.agents.hydra.VocabEncoder import VocabEncoder
from navsim.agents.hydra.PlanningVocabulary import PlanningVocabulary

from navsim.agents.hydra.VocabEmbedding import MLPEmbedder


class HydraModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: TransfuserConfig, device):
        """
        Initializes TransFuser torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()
        self.device = device
        #self._query_splits = [
        #    1,
        #    config.num_bounding_boxes,
        #]

        self._config = config
        self._backbone = TransfuserBackbone(config)
        
        #without imitation score-------------------------------------------------
        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._trajectory_head = TrajectoryHead(
            num_poses=trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(
                    config.lidar_resolution_height // 2,
                    config.lidar_resolution_width,
                ),
                mode="bilinear",
                align_corners=False,
            ),
        )
        #without imitation score-------------------------------------------------

        #with imitation score----------------------------------------------------
        #create Vocab
        #self.pv = PlanningVocabulary()
        #self.pv.createTrajectories()
        #self.pv.readTrajectories()
        #self.pv.selectKMeans()
        #self.pv.visualizeTrajectories()

        # Vocabulary Processing
        #self.status_embedder = MLPEmbedder(input_dim=8, hidden_dim=512, output_dim=config.tf_d_model)
        #self.vocab_embedder = MLPEmbedder(input_dim=24, hidden_dim=512, output_dim=config.tf_d_model)
        #self.vocab_encoder = VocabEncoder(embed_dim=config.tf_d_model, num_heads=1, num_layers=1)
        #self.query_proj = nn.Linear(512, config.tf_d_model)

        #Process for decoder -> create Q,K,V
        #self.keyval_proj = nn.Linear(8**2, config.tf_d_model)  
        #self._keyval_embedding = nn.Embedding(8**2, config.tf_d_model)  # 8x8 feature grid + trajectory
        #self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        #self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1) # Fenv
        #self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        
        # Transformer Decoder
        #tf_decoder_layer = nn.TransformerDecoderLayer(
        #    d_model=config.tf_d_model,
        #    nhead=config.tf_num_head,
        #    dim_feedforward=config.tf_d_ffn,
        #    dropout=config.tf_dropout,
        #    batch_first=True,
        #)
        #self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)


        #Head for imitation scoring -> later mutliple heads
        #self._imitation_head = ImitationHead(
        #    vocab_size = len(self.pv.anchor_poses),
        #    d_ffn=config.tf_d_ffn,
        #    d_model=config.tf_d_model,
        #    device = self.device
        #)
        #with imitation score----------------------------------------------------




        self.to(device)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]
        
        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        #without imitation score-------------------------------------------------
        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, _ = query_out.split(self._query_splits, dim=1)


        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)
        #without imitation score-------------------------------------------------

        #with imitation score----------------------------------------------------
        #transform feature vector to K,V for decoder
        #keyval = bev_feature  # (batch, N, 512)

        #Vocab for query
        #Vk = np.array(self.pv.anchor_poses)
        #Vk_flat = Vk.reshape(Vk.shape[0], -1)  # (k, 3*8)
        #Vk_tensor = torch.tensor(Vk_flat, dtype=torch.float32, device=self.device, requires_grad=False)
        #Vk_embedded = self.vocab_embedder(Vk_tensor)

        #
        #query = (Vk_embedded).expand(batch_size, -1, -1)
        #decoder_output = self._tf_decoder(query, keyval)

        

        #output.update({"trajectory": torch.tensor(self.pv.anchor_poses, dtype=torch.float32).to(self.device)})
        #imitation_score = self._imitation_head(decoder_output)
        #output.update(imitation_score)
        #with imitation score----------------------------------------------------

        return output


class ImitationHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, vocab_size: int, d_ffn: int, d_model: int, device):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(ImitationHead, self).__init__()

        self._vocab_size = vocab_size
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.device = device
        #self._mlp = nn.Sequential(
        #    nn.Linear(self._d_model, 1),
        #)
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.LayerNorm(self._d_ffn),
            nn.Linear(self._d_ffn, self._d_ffn // 2),
            nn.ReLU(),
            nn.LayerNorm(self._d_ffn // 2),
            nn.Linear(self._d_ffn // 2, 1),
        )



    def forward(self, decoder_output) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        imitation_scores = self._mlp(decoder_output)
        return {"imitation_score": imitation_scores}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
