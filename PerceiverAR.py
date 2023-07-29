import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    context_pdrop = 0.25
    perceiver = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

class PerceiverAR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.padding_idx)
        self.pos_emb = PositionalEmbedding(config)
        self.down_block = DownProjectBlock(config)
        self.blocks = nn.Sequential(
                        *[Block(config) for _ in range(config.n_layer - 1)]
                    )
        self.pred = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        x = self.emb(x) + self.pos_emb(x)
        x = self.down_block(x)
        x = self.blocks(x)
        x = self.pred(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(config.block_size, config.n_embd))

    def forward(self, x):
        batch_size, seq_len = x.shape
        all_outputs = []

        for i in range(batch_size):
            first_index = torch.nonzero(x[i], as_tuple=False)[0][0]
            kept_values = self.pos[:seq_len-first_index, :]

            # Create a tensor of zeros with the same shape as the positional embeddings that will be replaced
            zeros_to_add = torch.zeros([first_index, self.pos.size(1)], dtype=torch.float32, device = self.pos.device)

            # Replace the beginning of the positional embeddings with zeros
            output = torch.cat((zeros_to_add, kept_values), dim=0)
            all_outputs.append(output)

        return torch.stack(all_outputs, dim=0)

class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):
    def __init__(self, config, causal_mask = True, context = True):
        super().__init__()
        self.n_head = config.n_head
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.causal_mask = causal_mask
        self.context = context
        self.context_pdrop = config.context_pdrop
    
    def forward(self, x_k, x_v, x_q):

        # Batch, sequence length, embedding size
        B_k, T_k, C_k = x_k.size() 
        B_v, T_v, C_v = x_v.size()
        B_q, T_q, C_q = x_q.size()


        if self.context:
            # Randomly drop the context part of the key
            context_length = T_k - T_q
            rand = torch.zeros((B_k, context_length), device=x_k.device).uniform_()
            keep_context_len = context_length - int(context_length * self.context_pdrop)
            keep_indices = rand.topk(keep_context_len, dim = -1).indices
            keep_mask = torch.zeros_like(rand, device=x_k.device).scatter_(1, keep_indices, 1).bool()
            keep_mask_3d = keep_mask.unsqueeze(-1).expand(-1, -1, C_k)

            x_k_context = x_k[:, :context_length, :]
            x_k_context = torch.masked_select(x_k_context, keep_mask_3d).view(B_k, -1, C_k)
            x_k = torch.cat((x_k_context, x_k[:, context_length:, :]), dim = 1)
            B_k, T_k, C_k = x_k.size() 

            x_v_context = x_v[:, :context_length, :]
            x_v_context = torch.masked_select(x_v_context, keep_mask_3d).view(B_k, -1, C_k)
            x_v = torch.cat((x_v_context, x_v[:, context_length:, :]), dim = 1)
            B_v, T_v, C_v = x_v.size()

        k = self.key(x_k)
        v = self.value(x_v)
        q = self.query(x_q)

        # Split into heads and swap heads and sequence length
        k = k.view(B_k, T_k, self.n_head, C_k // self.n_head).transpose(1, 2) 
        q = q.view(B_q, T_q, self.n_head, C_q // self.n_head).transpose(1, 2)
        v = v.view(B_v, T_v, self.n_head, C_v // self.n_head).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, M, hs) x (B, nh, hs, L) -> (B, nh, M, L)

        if self.causal_mask:
            # Attention is a rectangular matrix with query on the rows and key on the columns
            # We want to apply a causal mask to the right of the rectangle as the query is the last elements of the key
            causal_mask = torch.ones_like(attn, dtype = torch.bool).triu(attn.size(-1) - attn.size(-2) + 1)
            attn = attn.masked_fill(causal_mask, -1e10)

        attn = self.attn_drop(attn)
        attn = F.softmax(attn, dim=-1)
        
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C_v)
        y = self.proj(y)
        return y

class DownProjectBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_size = config.latent_size
        self.attn = Attention(config, context = True)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        self.attn_sub_layer = SubLayerConnection(config.n_embd, config.resid_pdrop)
        self.mlp_sub_layer = SubLayerConnection(config.n_embd, config.resid_pdrop)

    def forward(self, x):
        downsize_x = x[:, -self.latent_size:, :] # Query only the latest tokens
        x = self.attn_sub_layer(downsize_x, lambda downsize_x: self.attn(x, x, downsize_x)) 
        x = self.mlp_sub_layer(x, self.mlp)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(4 * config.n_embd, config.n_embd),
            
        )
        self.attn_sub_layer = SubLayerConnection(config.n_embd, config.resid_pdrop)
        self.mlp_sub_layer = SubLayerConnection(config.n_embd, config.resid_pdrop)

    def forward(self, x):
        x = self.attn_sub_layer(x, lambda x: self.attn(x, x, x))
        x = self.mlp_sub_layer(x, self.mlp)
        return x