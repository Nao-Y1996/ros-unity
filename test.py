# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from load_data import LoadData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import csv

def s(_data):
    print(type(_data),np.shape(_data))


load_data = LoadData()
load_data.load()

# データ読み込み
# iris = datasets.load_iris()
# data = iris.data
# target = iris.target
target = load_data.target
data = load_data.data
with open('_target.csv', 'w') as f:
    writer = csv.writer(f)
    for i in target:
        writer.writerow([i])
with open('_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(data, target, shuffle=True)


# 特徴量の標準化
# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train = scaler.transform(x_train)
# x_valid = scaler.transform(x_valid)

# Tensor型に変換
# 学習に入れるときはfloat型 or long型になっている必要があるのここで変換してしまう
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
x_valid = torch.from_numpy(x_valid).float()
y_valid = torch.from_numpy(y_valid).long()

print('x_train : ', x_train.shape)
print('y_train : ', y_train.shape)
print('x_valid : ', x_valid.shape)
print('y_valid : ', y_valid.shape)

# 入力データと出力データのセットを作る
train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)

# 動作確認
# indexを指定すればデータを取り出すことができます。
# index = 0
# print(train_dataset.__getitem__(index)[0].size())
# print(train_dataset.__getitem__(index)[1])


batch_size = 30
# 「入力データと出力データのセット」からミニバッチを作成
# （batch_dize個で１つのミニバッチ、ミニバッチの個数はデータノサイズ/batich_size）
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# print(len(train_dataloader))
# for x_train, y_train in train_dataloader:
#     s(x_train)
#     s(y_train)

# 動作確認
# こんな感じでバッチ単位で取り出す子ができます。
# イテレータに変換
# batch_iterator = iter(train_dataloader)
# 1番目の要素を取り出す
# inputs, labels = next(batch_iterator)
# print(inputs.size())
# print(labels.size())


n_in = load_data.n_in
n_out = load_data.n_out
class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_in*3)
        self.fc2 = nn.Linear(n_in*3, n_out)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# エポック数
num_epochs = 100
# 1epoch ですべての学習用データを1回回したことになる
for epoch in range(num_epochs): #学習回数500回
    total_loss = 0
    count = 0
    for x_train, y_train in train_dataloader:
        count += 1
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() #loss.data[0]で記述するとPyTorch0.5以上ではエラーが返る
    
    # if (epoch+1)%50 == 0:
        # print(count, epoch+1, total_loss)
# print('-----検証-----')
# print(model(x_valid))
# result = torch.max(model(x_valid).data, 1)[1]
# # print(result)
# accuracy = sum(y_valid.data.numpy() == result.numpy()) / len(y_valid.data.numpy())
# print('{:.3f}'.format(accuracy))