import torch
import torch.nn as nn

class VolFormer(nn.Module):
    """
    Encoder for sequence-to-scalar task.

    Features:
    - Learnable token embeddings via a linear layer from d_in -> d_model.
    - Learnable positional embeddings (nn.Embedding).
    - Prepend[CLS] token for pooled representation.
    - Manual unrolling of TransformerEncoderLayer to expose per-layer attention maps.

    Parameters:
    d_in : int
        Input feature dimension per timestep.
    d_model : int, default=128
        Transformer hidden size / embedding dimension.
    nhead : int, default=4
        Number of attention heads.
    num_layers : int, default=3
        Number of Transformer encoder layers.
    p_drop : float, default=0.1
        Dropout rate used inside encoder layers and head.
    use_cls : bool, default=True
        If True, prepend a learnable [CLS] token and pool from it
    ff_mult : int, default=4 (suggested in Vaswani et al., 2017)
        Multiplier for the feed-forward dimension (ffn_dim = ff_mult * d_model).
    max_len : int, default=4096
        Maximum supported sequence length (without the CLS). Sequences longer than
        this will raise a ValueError.

    Inputs:
    x : torch.Tensor
        Shape [B, L, d_in]. Batch of sequences.

    Returns:
    output : torch.Tensor
        Shape [B]. Regression output per sequence.
    (output, attn) : Tuple[torch.Tensor, torch.Tensor] if return_attention=True
        - output: [B]
        - attn: [num_layers, B, nhead, L+1, L+1] +1 for CLS
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        p_drop: float = 0.1,
        use_cls: bool = True,
        ff_mult: int = 4,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.use_cls = use_cls
        self.d_model = d_model
        self.embed = nn.Linear(d_in, d_model)

        # positional embedding for tokens (add +1 for CLS position)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

        # define encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
              d_model=d_model, nhead=nhead,
              dim_feedforward=ff_mult*d_model,
              dropout=p_drop, batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        ])

        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

        # define heads
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p_drop),
            nn.Linear(d_model, 1)
        )

    # define forward pass
    def forward(self, x, return_attention=False):  # x: [B, L, d_in]
        B, L, _ = x.shape
        h = self.embed(x)

        # add positional embeddings
        # if CLS is used, its position index = 0, the rest shifted by +1
        if self.use_cls:
            pos_idx = torch.arange(1, L+1, device=x.device).unsqueeze(0).expand(B, -1)
            h = h + self.pos_emb(pos_idx)
            cls = self.cls.expand(B, -1, -1)

            # CLS gets position 0
            h = torch.cat([cls, h], dim=1)  # [B, L+1, d_model]
            h = h + torch.cat([self.pos_emb(torch.zeros(B,1, dtype=torch.long, device=x.device)),
                               torch.zeros_like(h[:,1:])], dim=1)
        else:
            pos_idx = torch.arange(0, L, device=x.device).unsqueeze(0).expand(B, -1)
            h = h + self.pos_emb(pos_idx)

        # manual encoder forward pass to get attention, only on test sets
        all_attention_maps = []
        for i, layer in enumerate(self.encoder_layers):
            # norm first before attention
            h_norm = layer.norm1(h)

            # preserve heads in attention map
            attn_output, attn_weights = layer.self_attn(
                h_norm, h_norm, h_norm,
                need_weights=True,
                average_attn_weights=False
            )
            h = h + layer.dropout1(attn_output)

            all_attention_maps.append(attn_weights)

            # feed-forward block
            h_norm = layer.norm2(h)
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(h_norm)))
                )
            h = h + layer.dropout2(ff_output)

        pooled = h[:, 0] if self.use_cls else h.mean(dim=1)
        output = self.head(pooled).squeeze(-1)

        if return_attention:
          # stack into [num_layers, B, n_heads, L+1, L+1]
            return output, torch.stack(all_attention_maps, dim=0)
        return output
