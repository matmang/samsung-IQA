import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import warnings
import argparse

from models.models import TransformerModel

warnings.filterwarnings(action='ignore')

# 설정
CFG = {
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'EPOCHS': 3,
    'LR': 0.001,
    'WEIGHT_DECAY': 0.0001,
    'GRADIENT_CLIP': 5.0
}

# ... (다른 필요한 모듈 임포트)
transform = transforms.Compose([
    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])), 
    transforms.ToTensor()
])

parser = argparse.ArgumentParser(description='DDP example')
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

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

train_data = pd.read_csv('../data/train.csv')

# 단어 사전 생성
all_comments = ' '.join(train_data['comments']).split()
vocab = set(all_comments)
vocab = ['<PAD>', '<SOS>', '<EOS>'] + list(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

test_data = pd.read_csv('../data/test.csv')
test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model_path = 'saved_model.pth'
model = TransformerModel(len(vocab)).cuda() 

state_dict = torch.load(model_path)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 'module.' 접두사 제거
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

model.eval()

def greedy_decode(model, img):
    with torch.no_grad():
        img = img.unsqueeze(0)  # batch_size 차원 추가
        mos_pred, captions_pred = model(img)

        print(mos_pred)
        print(captions_pred)

        # 아래와 같이 캡션 예측 결과를 얻습니다.
        predicted = captions_pred.argmax(2)
        tokens = predicted[0].cpu().numpy()

        # 토큰을 실제 단어로 디코드합니다.
        caption = []
        for token in tokens:
            word = idx2word[token]
            if word == '<EOS>':
                break
            if word != '<SOS>' and word != '<PAD>':
                caption.append(word)
        decoded_caption = ' '.join(caption)

    return mos_pred.squeeze().item(), decoded_caption



# 평가 함수
def evaluate(test_loader, model):
    predicted_mos_list = []
    predicted_comments_list = []
    predicted_comment = []

    with torch.no_grad():
        for images, _, _ in tqdm(test_loader):
            images = images.float().cuda(args.local_rank, non_blocking=True)

            mos_pred, full_captions_pred = model(images)
            for m in mos_pred.cpu().squeeze().tolist():
                predicted_mos_list.append(m) 
    
            if full_captions_pred is not None:
                for full_caption in full_captions_pred:
                    comment = ' '.join([idx2word[i.item()] for i in full_caption if i != word2idx['<PAD>']])
                    predicted_comments_list.append(comment)
            else:
                # 예측되지 않은 코멘트에 대해 기본 값을 추가
                for _ in range(images.size(0)):
                    predicted_comments_list.append("Nice Image.")

    return predicted_mos_list, predicted_comments_list


# 평가 실행
predicted_mos_list, predicted_comments_list = evaluate(test_loader, model)

print("comments: ",len(predicted_comments_list))
# print(predicted_comments_list)
print("mos: ", len(predicted_mos_list))

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list,
    'comments': predicted_comments_list
})

# 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv와 동일하게)
result_df['comments'] = result_df['comments'].fillna('Nice Image.')
result_df.to_csv('submit.csv', index=False)

print("Inference completed and results saved to submit.csv.")
