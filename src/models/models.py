# models.py
import torch
import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(BaseModel, self).__init__()

        # Image feature extraction using ResNet50
        self.cnn_backbone = models.resnet50(pretrained=True)
        # Remove the last fully connected layer to get features
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Image quality assessment head
        self.regression_head = nn.Linear(2048, 1)  # ResNet50 last layer has 2048 features

        # Captioning head
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2048, hidden_dim)  # Image features and caption embeddings as input
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        # CNN
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        # Image quality regression
        mos = self.regression_head(features_flat)

        # LSTM captioning
        if captions is not None:
            embeddings = self.embedding(captions)
            # Concatenate image features and embeddings for each word in the captions
            combined = torch.cat([features_flat.unsqueeze(1).repeat(1, embeddings.size(1), 1), embeddings], dim=2)
            lstm_out, _ = self.lstm(combined)
            outputs = self.fc(lstm_out)
            return mos, outputs
        else:
            return mos, None

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
        x = self.patch_emb(x).flatten(2).transpose(1, 2)
        x += self.position_emb
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.transformer.encoder(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transformer = nn.Transformer(emb_dim, num_heads, num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, enc_output):
        x = self.embedding(x)
        x = self.transformer.decoder(tgt=x, memory=enc_output)
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
    

class TransformerCaptioning(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(TransformerCaptioning, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.feature_proj = nn.Linear(2048, d_model)  # 가정: features_flat의 차원이 2048이라 가정하고, d_model로 투사합니다.
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, img_features, captions):
        src = self.feature_proj(img_features).permute(1, 0, 2)  # features를 d_model 차원으로 투사
        tgt = self.embedding(captions).permute(1, 0, 2)
        transformer_output = self.transformer(src, tgt)
        return self.fc(transformer_output.permute(1, 0, 2))


class TransformerModel(BaseModel):
    def __init__(self, vocab_size):
        super(TransformerModel, self).__init__(vocab_size)
        
        # Replace LSTM captioning head with Transformer
        d_model = 256  # This should match with the embedding dim
        self.captioning_head = TransformerCaptioning(vocab_size, d_model)

    def forward(self, images, captions=None):
        # CNN
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        # Image quality regression
        mos = self.regression_head(features_flat)

        # Transformer captioning
        if captions is not None:
            outputs = self.captioning_head(features_flat.unsqueeze(1), captions)
            return mos, outputs
        else:
            return mos, None
