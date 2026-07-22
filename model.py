import torch
import torch.nn as nn
import config as cfg


class LLaDAConfig:
    def __init__(self, 
                 vocab_size=cfg.TOKENIZER_VOCAB_SIZE, 
                 d_model=cfg.D_MODEL, 
                 n_layers=cfg.N_LAYERS, 
                 n_heads=cfg.N_HEADS, 
                 d_ff=cfg.D_FF, 
                 max_seq_len=cfg.MAX_SEQ_LEN):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len


class LLaDA(nn.Module):
    """
    LLaDA: Large Language Diffusion with mAsking.

    Architecture: Bidirectional Transformer encoder.
    Predicts p(x0 | xt) — the clean token given a masked input.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = LLaDAConfig()

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )
        
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """
        Unified initialization (BERT / GPT convention):
        - All weights: Normal(0, 0.02)
        - All biases: zero
        This applies to Embedding, Linear (attention + FFN), and output projection.
        """
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    module.weight.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.Linear):
                    module.weight.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, x):
        """x: (batch, seq_len) token ids  →  logits: (batch, seq_len, vocab_size)"""
        batch_size, seq_len = x.shape

        x = self.embedding(x)                                          # (B, L, D)
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x = x + self.pos_embedding(positions)
        x = self.transformer_encoder(x)
        logits = self.output_projection(x)                
        return logits



