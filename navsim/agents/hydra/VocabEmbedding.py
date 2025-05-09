import torch
import torch.nn as nn
import numpy as np

#pv = Vocab.PlanningVocabulary()
#pv.readTrajectories()
#pv.selectKMeans()

#Vk = np.array(Vocab.pv.anchors)
#Vk_flat = Vk.reshape(Vk.shape[0], -1)  # (k, 3*8)

class MLPEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    

    def forward(self, x):
        return self.mlp(x)


#mlp_embedder = MLPEmbedder(input_dim = Vk_flat.shape[1], hidden_dim=1024, output_dim=512)
#Vk_embedded = mlp_embedder(torch.tensor(Vk_flat, dtype=torch.float32))  # (k, 512)
#print(Vk_embedded)
#print(Vk_embedded.shape)
