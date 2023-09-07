import torch
import torch.nn as nn
import torchvision.models as models

from models.captioning_models import PureT

class IQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, max_caption_length=30):
        super(IQAModel, self).__init__()

        # ResNet50을 사용하여 이미지 특성 추출
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) # 마지막 분류 층 제거

        # 이미지 특성을 활용하여 화질 평가 점수 (mos) 예측
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Sigmoid를 사용하여 0~1 범위의 점수를 예측하게 함 (이후에 10을 곱해줄 예정)
        )

        # PureT Model Part (이미지 캡셔닝 관련)
        self.img_embedding = nn.Linear(2048, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(embed_dim, nhead=8, num_encoder_layers=3)  # 예시 파라미터 값입니다.
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_caption_length = max_caption_length

    def forward(self, x, captions=None):
        # IQA Model Prediction
        img_features = self.resnet(x)
        img_features_flatten = img_features.view(img_features.size(0), -1)
        mos = self.regression_head(img_features_flatten) * 10.0
    
        # 차원을 조정합니다. 
        img_features_embedded = self.img_embedding(img_features_flatten)
        img_features_permuted = img_features_embedded.unsqueeze(1).permute(1, 0, 2)
        
        # PureT Model Prediction
        if captions is not None:
            embedded_captions = self.embedding(captions)
            transformer_outputs = self.transformer(img_features_permuted, embedded_captions.permute(1, 0, 2))
            caption_predictions = self.fc_out(transformer_outputs.permute(1, 0, 2))
            return mos, caption_predictions
        
        else:
            generated_captions = self.greedy_decoding(img_features)
            return mos, generated_captions

        return mos, None

    def greedy_decoding(self, img_features):
        input_sequence = torch.ones((img_features.size(0), 1), dtype=torch.long).cuda()  # Assuming model is on CUDA
        for _ in range(self.max_caption_length):
            embedded_captions = self.embedding(input_sequence)
            transformer_outputs = self.transformer(img_features.permute(1, 0, 2), embedded_captions.permute(1, 0, 2))
            next_word_predictions = self.fc_out(transformer_outputs[:, -1, :])
            _, next_word = next_word_predictions.max(1, keepdim=True)
            input_sequence = torch.cat([input_sequence, next_word], dim=1)
        return input_sequence
