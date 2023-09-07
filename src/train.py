import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import random
import warnings

from models.iqa_models import IQAModel
import argparse
import torch.distributed as dist
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings(action='ignore')

# 설정
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LR': 0.001,
    'WEIGHT_DECAY': 0.0001,
    'GRADIENT_CLIP': 5.0
}

# 데이터 로드
train_data = pd.read_csv('../data/train.csv')

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

# CustomDataset 코드 필요
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # mos column 존재 여부에 따라 값을 설정
        mos = float(self.dataframe.iloc[idx]['mos']) if 'mos' in self.dataframe.columns else 0.0
        comment = self.dataframe.iloc[idx]['comments'] if 'comments' in self.dataframe.columns else ""
        
        return img, mos, comment

train_dataset = CustomDataset(train_data, transform)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True)

# DDP를 위한 설정
parser = argparse.ArgumentParser(description='DDP example')
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.backends.cudnn.benchmark = True

# 모델 및 옵티마이저 설정
model = IQAModel(len(vocab))
model = model.cuda(args.local_rank)
model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)

criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY'])

# 학습 함수
def train(train_loader, model, mos_criterion, caption_criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0

    # tqdm 객체 생성. position=0으로 설정하면 출력이 겹치지 않음. local_rank==0인 경우에만 출력
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True) if args.local_rank == 0 else enumerate(train_loader)
    for i, (images, mos, comments) in progress_bar:
        images, mos = images.float().cuda(args.local_rank, non_blocking=True), mos.float().cuda(args.local_rank, non_blocking=True)
        optimizer.zero_grad()

        # Convert comments to tensor
        comments_idx = [torch.Tensor([word2idx[word] for word in comment.split()]) for comment in comments]
        comments_tensor = nn.utils.rnn.pad_sequence(comments_idx, batch_first=True).long()
        comments_tensor = comments_tensor.cuda(args.local_rank, non_blocking=True)

        mos_pred, captions_pred = model(images, comments_tensor)

        # Loss calculation: sum of mos loss and captioning loss
        mos_loss = mos_criterion(mos_pred.squeeze(), mos)
        caption_loss = caption_criterion(captions_pred.view(-1, len(vocab)), comments_tensor.view(-1))
        loss = mos_loss + caption_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['GRADIENT_CLIP'])
        optimizer.step()

        total_loss += loss.item()

        # tqdm 설명 업데이트 (args.local_rank == 0일 때만)
        if args.local_rank == 0:
            avg_loss = total_loss / (i+1)
            progress_bar.set_description(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")


    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# 학습 실행
for epoch in range(CFG['EPOCHS']):
    train(train_loader, model, criterion1, criterion2, optimizer, epoch)

# 체크포인트 저장
if args.local_rank == 0:
    torch.save(model.module.state_dict(), "combined_model_checkpoint.pth")

print("Training finished!")