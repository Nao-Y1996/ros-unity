# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import LoadData
import numpy as np
from torch.utils.data import TensorDataset

# デバッグ用
def info(string, _data):
    print(string + '=> 型：'+str(type(_data))+', 形状：'+str(np.shape(_data)))

# データの読み込み
data = LoadData()
data.load()
_DIR_HERE =  data.dir_here
# 学習データと検証データに分割
x_train, x_valid, y_train, y_valid = train_test_split(data.input_train, data.correct_train, shuffle=True)

input_data = torch.from_numpy(data.input_train.astype(np.float32))
correct_data = torch.from_numpy(data.correct_train.astype(np.float32))
info('input_data', input_data)
info('correct_data', correct_data)
print(correct_data[:100])

#NNの定義
n_in = np.shape(data.input_train)[1]
n_mid = n_in * 4
n_out = data.n_out
model = nn.Sequential(
    nn.Linear(n_in,n_mid),
    nn.Sigmoid(),
    nn.Linear(n_mid,n_out)
)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    y = model(input_data)
    print(np.shape(y))
    print(y[:100])
    loss = loss_func(y, correct_data)
    loss.backward()
    optimizer.step()

print(model(input_data)[0:100])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ml = nn.Linear(n_in,n_mid)
        self.ol = nn.Linear(n_mid, n_out)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    def forward(self, x):
        x = self.ml(x)
        x = nn.functional.relu(x)
        y = self.ol(x)
        return output