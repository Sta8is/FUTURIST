# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
import einops
from torch import nn


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x, mask=None):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        if mask is not None:
            attention_value, attention_weight = self.mha(x, x, x, attn_mask=mask)
        else:
            attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, full_shape=None):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn


class TransformerEncoderSeperableAttention(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., window_size=1):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, full_shape):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        b, t, h, w, d = full_shape
        l_attn = []
        for attn_temporal, attn_spatial, ff in self.layers:
            if self.window_size == 1:
                x = einops.rearrange(x, 'b (t h w) c -> (b h w) t c', b=b, t=t, h=h, w=w)
            else:
                x = einops.rearrange(x, 'b (t h w) c -> b t h w c', b=b, t=t, h=h, w=w)
                x = einops.rearrange(
                    x, 'b t (h k1) (w k2) c -> (b h w) (t k1 k2) c',
                    k1=self.window_size, k2=self.window_size)
            # mask_temp = torch.triu(torch.ones(t, t),diagonal=1).masked_fill(torch.arange(t).unsqueeze(0) < t-3, 0).to(x.device)
            # attention_value, attention_weight = attn_temporal(x, mask=mask_temp)
            attention_value, attention_weight = attn_temporal(x)
            x = x + attention_value
            l_attn.append(attention_weight)

            if self.window_size == 1:
                x = einops.rearrange(x, '(b h w) t c -> (b t) (h w) c', b=b, t=t, h=h, w=w)
            else:
                x = einops.rearrange(
                    x, '(b h w) (t k1 k2) c -> (b t) (h k1) (w k2) c',
                    b=b, t=t, h=h//self.window_size, w=w//self.window_size, k1=self.window_size, k2=self.window_size)
                x = x.flatten(1,2)
            attention_value, attention_weight = attn_spatial(x)
            x = x + attention_value
            l_attn.append(attention_weight)

            x = einops.rearrange(x, '(b t) (h w) c -> b (t h w) c', b=b, t=t, h=h, w=w)
            x = x + ff(x)

        return x, l_attn


class MaskTransformer(nn.Module):
    def __init__(self, shape, img_size=256, embedding_dim = 768, hidden_dim=768,  depth=24, heads=8, mlp_dim=3072, 
                dropout=0.1, use_fc_bias=False, use_first_last=False, seperable_attention=False, seperable_window_size=1, sep_head = False):
        super().__init__()
        self.pos_embd = AddBroadcastPosEmbed(shape=shape, embd_dim=hidden_dim)
        self.sep_head = sep_head
        # First layer before the Transformer block
        self.first_layer = nn.Identity() 
        if use_first_last:
            self.first_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim), # hidden_dim = 768
            )

        self.seperable_attention = seperable_attention
        Transformer = TransformerEncoderSeperableAttention if seperable_attention else TransformerEncoder
        self.transformer = Transformer(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if seperable_attention:
            self.transformer = TransformerEncoderSeperableAttention(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                window_size=seperable_window_size)
        else:
            self.transformer = TransformerEncoder(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Identity()
        self.last_layer_segm = nn.Identity()
        self.last_layer_depth = nn.Identity()
        if use_first_last:
            if not sep_head:
                self.last_layer = nn.Sequential(
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                )
            else:
                self.last_layer_segm = nn.Sequential(
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                )
                self.last_layer_depth = nn.Sequential(
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim, eps=1e-12),
                )

        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=use_fc_bias)
        self.fc_in.weight.data.normal_(std=0.02) 
        if not sep_head:
            self.fc_out = nn.Linear(hidden_dim, embedding_dim, bias=use_fc_bias)
            self.fc_out.weight.data.copy_(torch.zeros(embedding_dim, hidden_dim))
        else:
            output_segmsize = int(0.5*embedding_dim)
            self.fc_out_segm = nn.Linear(hidden_dim, output_segmsize, bias=use_fc_bias)
            self.fc_out_segm.weight.data.copy_(torch.zeros(output_segmsize, hidden_dim))
            output_depthsize = int(0.5*embedding_dim)
            self.fc_out_depth = nn.Linear(hidden_dim, output_depthsize, bias=use_fc_bias)
            self.fc_out_depth.weight.data.copy_(torch.zeros(output_depthsize, hidden_dim))

    def forward(self, vid_token, y=None, drop_label=None, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, t, h, w, c = vid_token.size()

        # Position embedding
        vid_token = self.fc_in(vid_token)
        vid_token_embeddings = self.pos_embd(vid_token) 
        x = einops.rearrange(vid_token_embeddings, 'b t h w c -> b (t h w) c')

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, full_shape=[b, t, h, w, c])
        if not self.sep_head:
            x = self.last_layer(x)
            x_out = self.fc_out(x)
        else:
            x_segm = self.last_layer_segm(x)
            x_depth = self.last_layer_depth(x)
            x_segm = self.fc_out_segm(x_segm)
            x_depth = self.fc_out_depth(x_depth)
            x_out = torch.cat([x_segm, x_depth.to(x.device)], dim=-1)
        if return_attn:
            return x_out, attn
        else:
            return x_out

################ Spatiotemporal broadcasted positional embeddings ###############
class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape # [T, H/8, W/8]
        self.n_dim = n_dim = len(shape) 
        self.embd_dim = embd_dim 
        self.dim = dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ParameterDict({
            f'd_{i}': nn.init.trunc_normal_(nn.Parameter(torch.randn(shape[i], embd_dim // n_dim)),0.,0.02)
                                    if dim == -1 else
                                    nn.init.trunc_normal_(torch.randn(embd_dim // n_dim, shape[i]),0.,0.02)
            for i in range(n_dim)
        })

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        return x + embs