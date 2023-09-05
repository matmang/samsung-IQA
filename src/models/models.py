# models.py
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, emb_dim, num_heads, num_layers, num_classes):
        super(VisionTransformer, self).__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        self.patch_emb = nn.Conv2d(num_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.position_emb = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.transformer = nn.Transformer(emb_dim, num_heads, num_layers)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_emb(x).flatten(2).transpose(1, 2)  # Convert images to sequences of flattened patches
        x += self.position_emb
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer(x)
        x = self.fc(x[:, 0])
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transformer = nn.Transformer(emb_dim, num_heads, num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, enc_output):
        x = self.embedding(x)
        x = self.transformer(x, enc_output)
        x = self.fc(x)
        return x


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, emb_dim, num_heads, num_layers, num_classes, vocab_size):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = VisionTransformer(image_size, patch_size, num_channels, emb_dim, num_heads, num_layers, num_classes)
        self.decoder = TransformerDecoder(emb_dim, vocab_size, num_heads, num_layers)

    def forward(self, images, captions):
        enc_output = self.encoder(images)
        dec_output = self.decoder(captions, enc_output)
        return dec_output