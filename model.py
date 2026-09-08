import torch
import torch.nn as nn
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indices = np.arange(size)
    np.random.shuffle(forward_indices)
    backward_indices = np.argsort(forward_indices)
    return forward_indices, backward_indices


def take_indexes(sequences: torch.Tensor, indexes: torch.Tensor):
    # sequences: [T, B, C], indexes: [T, B] -> gathered: [T, B, C]
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    def __init__(self, ratio: float) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        # patches: [T, B, C]
        T, B, _ = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long, device=patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long, device=patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 12,
        num_head: int = 3,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patchify = nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, img: torch.Tensor):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, _, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        num_layer: int = 4,
        num_head: int = 3,
    ) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, features: torch.Tensor, backward_indexes: torch.Tensor):
        T = features.shape[0]

        # Shift backward indices to account for prepended [CLS] token
        cls_idx = torch.zeros(1, backward_indexes.shape[1], dtype=torch.long, device=backward_indexes.device)
        backward_indexes = torch.cat([cls_idx, backward_indexes + 1], dim=0)

        # Append learnable mask tokens to match full sequence length
        num_mask_tokens = backward_indexes.shape[0] - features.shape[0]
        mask_tokens = self.mask_token.expand(num_mask_tokens, features.shape[1], -1)
        features = torch.cat([features, mask_tokens], dim=0)

        # Unshuffle tokens to original spatial order and add decoder position embeddings
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove [CLS] token

        patches = self.head(features)

        # Generate binary mask aligned with original spatial positions (1 for masked, 0 for visible)
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)

        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        emb_dim: int = 192,
        encoder_layer: int = 12,
        encoder_head: int = 3,
        decoder_layer: int = 4,
        decoder_head: int = 3,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img: torch.Tensor):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes: int = 10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img: torch.Tensor):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        
        logits = self.head(features[0])  # classify via [CLS] token
        return logits


if __name__ == '__main__':
    img = torch.rand(2, 3, 32, 32)
    model = MAE_ViT()
    predicted_img, mask = model(img)
    loss = torch.mean((predicted_img - img) ** 2 * mask) / 0.75

    print("Input shape:      ", img.shape)
    print("Predicted shape:  ", predicted_img.shape)
    print("Mask shape:       ", mask.shape)
    print("Computed MSE loss:", loss.item())