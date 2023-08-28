import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import CFG
from datasets.custom_dataset import CustomDataset
from models.captioning_model import BaseModel

test_data = pd.read_csv('test.csv')
test_dataset = CustomDataset(test_data, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
predicted_mos_list = []
predicted_comments_list = []

def greedy_decode(model, image, max_length=50):
    image = image.unsqueeze(0).cuda()
    mos, _ = model(image)
    output_sentence = []
    
    # 시작 토큰 설정
    current_token = torch.tensor([word2idx['<SOS>']]).cuda()
    hidden = None
    features = model.cnn(image).view(image.size(0), -1)

    for _ in range(max_length):
        embeddings = model.embedding(current_token).unsqueeze(0)
        combined = torch.cat([features.unsqueeze(1), embeddings], dim=2)
        out, hidden = model.lstm(combined, hidden)
        
        output = model.fc(out.squeeze(0))
        _, current_token = torch.max(output, dim=1)

        # <EOS> 토큰에 도달하면 멈춤
        if current_token.item() == word2idx['<EOS>']:
            break

        # <SOS> 또는 <PAD> 토큰은 생성한 캡션에 추가하지 않음
        if current_token.item() not in [word2idx['<SOS>'], word2idx['<PAD>']]:
            output_sentence.append(idx2word[current_token.item()])
     
    return mos.item(), ' '.join(output_sentence)

# 추론 과정
with torch.no_grad():
    for imgs, _, _ in tqdm(test_loader):
        for img in imgs:
            img = img.float().cuda()
            mos, caption = greedy_decode(model, img)
            predicted_mos_list.append(mos)
            predicted_comments_list.append(caption)

# 결과 저장
result_df = pd.DataFrame({
    'img_name': test_data['img_name'],
    'mos': predicted_mos_list,
    'comments': predicted_comments_list  # 캡션 부분은 위에서 생성한 것을 사용
})

# 예측 결과에 NaN이 있다면, 제출 시 오류가 발생하므로 후처리 진행 (sample_submission.csv과 동일하게)
result_df['comments'] = result_df['comments'].fillna('Nice Image.')
result_df.to_csv('submit.csv', index=False)

print("Inference completed and results saved to submit.csv.")