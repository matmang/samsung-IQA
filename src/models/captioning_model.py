import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super(BaseModel, self).__init__()

        self.cnn_backbone = models.resnet50(pretrained=True)
        modules = list(self.cnn_backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        self.regression_head = nn.Linear(2048, 1)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 2048, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions=None):
        features = self.cnn(images)
        features_flat = features.view(features.size(0), -1)

        mos = self.regression_head(features_flat)

        if captions is not None:
            embeddings = self.embedding(captions)
            combined = torch.cat([features_flat.unsqueeze(1).repeat(1, embeddings.size(1), 1), embeddings], dim=2)
            lstm_out, _ = self.lstm(combined)
            outputs = self.fc(lstm_out)
            return mos, outputs
        else:
            return mos, None
