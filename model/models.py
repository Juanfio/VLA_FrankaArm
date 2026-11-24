import math

import torch
import torch.nn as nn

# Output dimension must be input dimension of language model (256 to match with OpenVLA model).

class PatchEmbedding(nn.Module):
    """
    Splits an image into fixed-size patches and linearly projects each to
    dimension D — Eq. 1 in the paper: x_p in R^(N x (P^2*C)) -> R^(N x D).

    Implemented as a single strided Conv2d, which is mathematically identical
    to "flatten each PxP patch, then apply a shared linear layer" but avoids
    the explicit flatten/unfold step: a Conv2d with kernel_size=stride=P applied
    to a non-overlapping patch produces exactly one linear projection per patch.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.num_patches = (img_size // patch_size) ** 2  # N = HW / P^2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)                     # (B, D, H/P, W/P)
        x = x.flatten(2)                     # (B, D, N)
        x = x.transpose(1, 2)                # (B, N, D)  -- patch embeddings, sequence-first
        return x


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder layer, pre-norm as specified in (Dosovitsky et al 2021, arXiv):
        z'_l = MSA(LN(z_{l-1})) + z_{l-1}      (Eq. 2)
        z_l  = MLP(LN(z'_l)) + z'_l            (Eq. 3)
    MLP is two linear layers with a GELU nonlinearity in between.
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- MSA(LN(x)) + residual ---
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # --- MLP(LN(x)) + residual ---
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Pure ViT encoder: patchify -> [class] token + positional embedding ->
    L transformer encoder blocks -> final LayerNorm -> image representation y.

    Follows Eq. 1-4:
        z0   = [x_class; x_p^1 E; ...; x_p^N E] + E_pos
        z'_l = MSA(LN(z_{l-1})) + z_{l-1},  l = 1..L
        z_l  = MLP(LN(z'_l)) + z'_l,        l = 1..L
        y    = LN(z_L^0)                     <- state of [class] token, final layer

    References:
        AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE - Dosovitskiy
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768, # Table 1 in Dosovitsky et al 2021, arXiv. Hidden size D, output of CNN.
        num_layers: int = 12, # Table 1 in Dosovitsky et al 2021, arXiv.
        num_heads: int = 12, # Table 1 in Dosovitsky et al 2021, arXiv.
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        use_lang: bool = False,
        lang_dim: int = 384
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.use_lang = use_lang
        self.embed_dim = embed_dim

        # learnable [class] token, prepended to the patch sequence (Section 3.1)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # standard learnable 1D position embeddings (paper found 2D-aware embeddings
        # gave no significant improvement, Appendix D.4) -- one per position,
        # including the [class] token position (hence num_patches + 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)  # Eq. 4: y = LN(z_L^0)

        # Language projection (only for vision-language mode)
        if use_lang:
            self.lang_proj = nn.Linear(lang_dim, embed_dim)
            nn.init.trunc_normal_(self.lang_proj.weight, std=0.02)
            nn.init.zeros_(self.lang_proj.bias)

        # fill a learnable parameter with small random values, drawn from a truncated normal distribution, before training starts.
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, lang: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, 3, H, W) -- a batch of preprocessed images.
        lang: (B, lang_dim) -- language embeddings (optional, only when use_lang=True).
        Returns: (B, D) -- the [class] token's final representation, y.
        """
        B = x.shape[0]

        x = self.patch_embed(x)                              # (B, N, D)

        class_tokens = self.class_token.expand(B, -1, -1)     # (B, 1, D)
        x = torch.cat([class_tokens, x], dim=1) + self.pos_embed # (B, N+1, D)

        if self.use_lang and lang is not None:
            lang_tok = self.lang_proj(lang).unsqueeze(1)       # (B, 1, D)
            x = torch.cat([x, lang_tok], dim=1)                # (B, N+2, D)

        x = self.dropout(x)

        for block in self.encoder_blocks:
            x = block(x)

        x = self.final_norm(x)                                  # Eq. 4

        return x[:, 0]  # y: representation of the [class] token, shape (B, D)


def _load_dinov2(model_name: str) -> nn.Module:
    """
    Load a pretrained DINOv2 backbone through torch.hub (cache: ~/.cache/torch/hub; compute nodes without
    internet need it populated beforehand). Under DDP rank 0 loads first, so ranks never download at once.
    """
    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized()
    is_rank0 = (not distributed) or dist.get_rank() == 0
    if not is_rank0:
        dist.barrier()  # wait until rank 0 has populated the cache
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    if distributed and is_rank0:
        dist.barrier()  # release the other ranks
    return model


class DinoV2Encoder(nn.Module):
    """
    Pretrained DINOv2 image encoder (Oquab et al. 2023) with VisionTransformer's interface:
    forward(images, lang) -> image representation. Images must be ImageNet-normalized (DINOv2's pretraining
    normalization) with sides divisible by the patch size 14 (224x224 -> 16x16 patches).
    Language is not injected into the backbone; with use_lang=True the ActionHead concatenates it.

    pooling: "cls" -> CLS token, (B, D); "cls_mean" -> CLS token + mean patch token, (B, 2D) (the feature
    DINOv2 uses for its linear-probe evaluations); "grid" -> CLS token followed by the patch tokens averaged
    over a grid_size x grid_size grid, (B, 1 + grid_size**2, D), which keeps where things are in the image.

    train_blocks: with freeze=True, the last train_blocks transformer blocks and the final LayerNorm still train
    (partial fine-tuning). The frozen blocks before them then run once, in preprocessing (see cache_features).
    """

    def __init__(self, model_name: str = "dinov2_vits14", freeze: bool = True, pooling: str = "cls",
                 use_lang: bool = False, grid_size: int = 4, train_blocks: int = 0):
        super().__init__()
        if pooling not in ("cls", "cls_mean", "grid"):
            raise ValueError(f"pooling must be 'cls', 'cls_mean' or 'grid', got {pooling!r}")

        self.backbone = _load_dinov2(model_name)
        depth = len(self.backbone.blocks)                               # 12 for ViT-S/14
        if not 0 <= train_blocks <= depth:
            raise ValueError(f"train_blocks must be between 0 and {depth}, got {train_blocks}")
        self.model_name = model_name
        self.freeze = freeze
        self.train_blocks = train_blocks if freeze else 0               # freeze=False trains every block
        self.n_frozen = depth - train_blocks if freeze else 0           # blocks that never train
        self.pooling = pooling
        self.grid_size = grid_size
        # identifies its cached features in preprocessed files
        self.name = (f"{model_name}-tokens_after_block{self.n_frozen}" if self.train_blocks
                     else f"{model_name}-{pooling}" + (str(grid_size) if pooling == "grid" else ""))
        self.use_lang = use_lang
        self.patch_size = self.backbone.patch_size
        self.feature_dim = self.backbone.embed_dim                      # 384 for ViT-S/14
        self.embed_dim = self.feature_dim * (2 if pooling == "cls_mean" else 1)  # size of each output vector
        self.n_tokens = 1 + grid_size ** 2 if pooling == "grid" else 1  # output vectors per image

        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.blocks[self.n_frozen:].requires_grad_(True)  # the last train_blocks blocks
            self.backbone.norm.requires_grad_(self.train_blocks > 0)
            self.backbone.eval()

    def train(self, mode: bool = True):
        # Keep the frozen part of the backbone in eval mode even when the surrounding model trains.
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
            self.backbone.blocks[self.n_frozen:].train(mode)
        return self

    def _frozen_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds images as patch tokens and runs them through the frozen blocks.

            Args:
                x: torch.Tensor
                    (B, 3, H, W) ImageNet-normalized images, H and W divisible by the patch size 14.

            Returns:
                tokens: torch.Tensor
                    (B, 1 + h*w, D) CLS and patch tokens after block n_frozen (all blocks when train_blocks=0).
        """
        H, W = x.shape[-2:]
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"Image size {H}x{W} is not divisible by patch size {self.patch_size}")
        # No activations are stored for backprop through frozen weights.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            x = self.backbone.prepare_tokens_with_masks(x)
            for block in self.backbone.blocks[:self.n_frozen]:
                x = block(x)
        return x

    @torch.no_grad()
    def cache_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes what preprocessing stores per frame, so that training only runs the trainable part.

            Args:
                x: torch.Tensor
                    (B, 3, H, W) ImageNet-normalized images, H and W divisible by the patch size 14.

            Returns:
                features: torch.Tensor
                    The encoder output when every block is frozen (as forward). With train_blocks > 0, the tokens
                    after the frozen blocks, (B, 1 + h*w, D) in float16 to halve memory.
        """
        return self._frozen_tokens(x).half() if self.train_blocks else self(x)

    def forward(self, x: torch.Tensor, lang: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, 3, H, W) ImageNet-normalized images, H and W divisible by 14, or what cache_features returned.
        lang: ignored here (see class docstring).
        Returns: (B, embed_dim), or (B, n_tokens, embed_dim) for pooling="grid".
        """
        if x.dim() != 4 and not self.train_blocks:
            return x                                                    # final features cached in preprocessing
        hw = (x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size) if x.dim() == 4 else None
        x = self._frozen_tokens(x) if x.dim() == 4 else x.float()       # (B, 1 + h*w, D)
        for block in self.backbone.blocks[self.n_frozen:]:              # trainable blocks
            x = block(x)
        x = self.backbone.norm(x)

        cls_token = x[:, 0]                                             # (B, D)
        if self.pooling == "cls":
            return cls_token
        patches = x[:, 1 + self.backbone.num_register_tokens:]          # (B, h*w, D), row by row
        if self.pooling == "cls_mean":
            return torch.cat([cls_token, patches.mean(dim=1)], dim=-1)  # (B, 2D)
        B, N, D = patches.shape
        h, w = hw or (math.isqrt(N),) * 2                               # cached tokens: square images
        grid = patches.transpose(1, 2).reshape(B, D, h, w)
        grid = nn.functional.adaptive_avg_pool2d(grid, self.grid_size).flatten(2).transpose(1, 2)  # (B, G*G, D)
        return torch.cat([cls_token[:, None], grid], dim=1)             # (B, 1 + G*G, D)


class AttentionPooling(nn.Module):
    """
    Pools tokens (B, T, D) into one vector (B, D): a learned query attends over the tokens, each tagged with a
    learned position embedding, so the head can weight image regions by their relevance to the action.
    """

    def __init__(self, dim: int, n_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=max(1, dim // 64), dropout=dropout, batch_first=True)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens + self.pos_embed)
        pooled, _ = self.attn(self.query.expand(len(x), -1, -1), x, x, need_weights=False)
        return pooled[:, 0]


class ActionHead(nn.Module):
    """
    Small MLP head mapping the image representation (y) to per-dimension logits over the n_bins
    discretized action bins -- i.e. classification, not regression. Dropout acts on the image features.
    With n_tokens > 1 (e.g. a grid of patch features), attention pooling first turns the tokens into one vector.

    Supports vision-only and vision-language fusion via concatenation.
    """

    def __init__(self, embed_dim: int, action_dim: int, n_bins: int = 32, hidden_dim: int = 512,
                 use_lang: bool = False, lang_dim: int = 384, dropout: float = 0.0, n_tokens: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.use_lang = use_lang
        self.pool = AttentionPooling(embed_dim, n_tokens, dropout) if n_tokens > 1 else None
        self.dropout = nn.Dropout(dropout)  # outside self.mlp: keeps the checkpoints' parameter names

        # Input dimension: vision + (language if enabled)
        input_dim = embed_dim + (lang_dim if use_lang else 0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * n_bins), # the output dimension matches that of the action space.
        )

    def forward(self, y: torch.Tensor, lang: torch.Tensor | None = None) -> torch.Tensor:
        # y: (B, D_vision) or (B, n_tokens, D_vision) -> (B, action_dim, n_bins) logits, one softmax per action dim
        if self.pool is not None:
            y = self.pool(y)
        x = self.dropout(y)
        if self.use_lang and lang is not None:  # concatenate the language embedding if enabled
            x = torch.cat([x, lang], dim=-1)  # (B, D_vision + D_lang)
        return self.mlp(x).view(y.shape[0], self.action_dim, self.n_bins)


class RoboticActionModel(nn.Module):
    """
    ViT image encoder + action head.
    One ActionHead per dataset. Action space differ per dataset.

    Supports:
    - Pure vision model: ViT encodes images only
    - Vision-language model: ViT encodes images + language tokens, fusion via concatenation
    """

    def __init__(self, vit: VisionTransformer | DinoV2Encoder, action_heads: dict[str, ActionHead]):
        super().__init__()
        self.vit = vit

        # Validate action_heads match vit's use_lang setting
        for name, head in action_heads.items():
            if head.use_lang != self.vit.use_lang:
                raise ValueError(
                    f"ActionHead '{name}' use_lang={head.use_lang} does not match "
                    f"ViT use_lang={self.vit.use_lang}. All action heads must match ViT mode."
                )

        self.action_heads = nn.ModuleDict(action_heads)

    def forward(self, images: torch.Tensor, lang: torch.Tensor, dataset_name: str) -> torch.Tensor:
        # images: (B, 3, H, W), or what a frozen DINOv2 encoder cached in preprocessing (it handles both)
        y = self.vit(images, lang)                          # shared image encoder
        return self.action_heads[dataset_name](y, lang)     # (B, action_dim, n_bins)