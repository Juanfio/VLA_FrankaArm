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


class ActionHead(nn.Module):
    """
    Small MLP head mapping the ViT's image representation (y) to an action
    vector. Given actions are already discretized to [0, 255] bins per
    dimension (from earlier preprocessing), this predicts per-dimension
    logits over 256 classes -- i.e. classification, not regression.

    Supports vision-only and vision-language fusion via concatenation.
    """

    def __init__(self, embed_dim: int, action_dim: int, n_bins: int = 32, hidden_dim: int = 512,
                 use_lang: bool = False, lang_dim: int = 384):
        super().__init__()
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.use_lang = use_lang

        # Input dimension: vision + (language if enabled)
        input_dim = embed_dim + (lang_dim if use_lang else 0)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * n_bins), # the output dimension matches that of the action space.
        )
        # print(f'ActionHead linear in features: {nn.Linear(input_dim, hidden_dim).in_features}')
        # print(f'ActionHead linear out features: {nn.Linear(input_dim, hidden_dim).out_features}')

    def forward(self, y: torch.Tensor, lang: torch.Tensor | None = None) -> torch.Tensor:
        # y: (B, D_vision) -> (B, action_dim, n_bins) logits, one softmax per action dim
        B = y.shape[0]

        # Concatenate language embedding if enabled
        if self.use_lang and lang is not None:
            x = torch.cat([y, lang], dim=-1)  # (B, D_vision + D_lang)
        else:
            x = y

        out = self.mlp(x)
        return out.view(B, self.action_dim, self.n_bins) # for a given action dimension, the probability of having a specific bin (from 0 - 255).


class RoboticActionModel(nn.Module):
    """
    ViT image encoder + action head.
    One ActionHead per dataset. Action space differ per dataset.

    Supports:
    - Pure vision model: ViT encodes images only
    - Vision-language model: ViT encodes images + language tokens, fusion via concatenation
    """

    def __init__(self, vit: VisionTransformer, action_heads: dict[str, ActionHead]):
        super().__init__()
        self.vit = vit

        # Validate action_heads match vit's use_lang setting
        for name, head in action_heads.items():
            if head.use_lang != self.vit.use_lang:
                raise ValueError(
                    f"ActionHead '{name}' use_lang={head.use_lang} does not match "
                    f"ViT use_lang={self.vit.use_lang}. All action heads must match ViT mode."
                )

        for name, child in self.vit.named_children():
            print("=====================")
            print(f"Name: {name} \nChild: {child}")
        self.action_heads = nn.ModuleDict(action_heads)

    def forward(self, images: torch.Tensor, lang: torch.Tensor, dataset_name: str) -> torch.Tensor:
        y = self.vit(images, lang)                          # (B, D) shared image encoder
        return self.action_heads[dataset_name](y, lang)     # (B, action_dim, n_bins)