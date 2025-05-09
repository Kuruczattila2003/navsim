import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, V_prime_k, Fenv):
        """
        V_prime_k: (seq_len, batch, embed_dim) -> Query
        Fenv: (seq_len, batch, 512) -> Key & Value
        """
        # Ensure Fenv matches embedding dimension (512 -> embed_dim)
        Fenv = torch.tensor(Fenv)
        print(Fenv.shape)
        Fenv = Fenv.unsqueeze(1)  # (batch, seq_len, 512) -> (seq_len, batch, 512)

        # Transformer decoder
        V_double_prime_k = self.transformer_decoder(V_prime_k, Fenv)

        return V_double_prime_k

def imitation_loss(predicted_logits, log_replay_trajectory, vocab):
    """
    Computes the distance-based imitation loss.
    
    predicted_logits: Output from decoder (softmax scores) -> (seq_len, batch, vocab_size)
    log_replay_trajectory: The ground truth trajectory from human data -> (seq_len, embed_dim)
    vocab: The trajectory vocabulary -> (vocab_size, seq_len, embed_dim)
    """
    vocab_torch = torch.tensor(vocab, dtype=torch.float32)  # (vocab_size, seq_len, embed_dim)
    log_replay_torch = torch.tensor(log_replay_trajectory, dtype=torch.float32)  # (seq_len, embed_dim)

    # Compute L2 distance between log-replay trajectory and each vocabulary trajectory
    distances = torch.norm(vocab_torch - log_replay_torch.unsqueeze(0), dim=(1, 2))  # (vocab_size,)

    # Compute probability distribution using softmax
    targets = F.softmax(-distances, dim=0)  # (vocab_size,)

    # Compute cross-entropy loss
    return F.cross_entropy(predicted_logits.squeeze(1), targets.unsqueeze(0))
