from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn as nn

class VocabEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


#transformer = VocabEncoder(embed_dim=Embedding.Vk_embedded[1], num_heads=4, num_layers=3)
#V_prime_k = transformer(Embedding.Vk_embedded.unsqueeze(1))  # (k, 1, 512) -> Transformer expects (seq_len, batch, embed_dim)
#V_prime_k = V_prime_k.squeeze(1)  # Back to (k, 512)
