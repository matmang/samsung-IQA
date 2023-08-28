import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore') 
from config import CFG
from datasets.custom_dataset import CustomDataset
from models.captioning_model import BaseModel
from utils.train_utils import seed_everything

# ... (The provided training code here)
# 데이터 로드
train_data = pd.read_csv('train.csv')

# 단어 사전 생성
all_comments = ' '.join(train_data['comments']).split()
vocab = set(all_comments)
vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# 데이터셋 및 DataLoader 생성
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor()
])
train_dataset = CustomDataset(train_data, transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

# 모델, 손실함수, 옵티마이저
model = BaseModel(len(vocab)).cuda()
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])

# 학습
model.train()
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    for imgs, mos, comments in loop:
        imgs, mos = imgs.float().cuda(), mos.float().cuda()
        
        # Batch Preprocessing
        comments_tensor = torch.zeros((len(comments), len(max(comments, key=len)))).long().cuda()
        for i, comment in enumerate(comments):
            tokenized = ['<SOS>'] + comment.split() + ['<EOS>']
            comments_tensor[i, :len(tokenized)] = torch.tensor([word2idx[word] for word in tokenized])

        # Forward & Loss
        predicted_mos, predicted_comments = model(imgs, comments_tensor)
        loss1 = criterion1(predicted_mos.squeeze(1), mos)
        loss2 = criterion2(predicted_comments.view(-1, len(vocab)), comments_tensor.view(-1))
        loss = loss1 + loss2

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1} finished with average loss: {total_loss / len(train_loader):.4f}")