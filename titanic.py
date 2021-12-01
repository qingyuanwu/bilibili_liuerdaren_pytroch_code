import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd

class Titanic(Dataset):
    def __init__(self, filepath):
        row_data = pd.read_csv(filepath)
        xy = np.float32(row_data)

        self.x_data = torch.from_numpy(xy[:, 1:])
        self.y_data = torch.from_numpy(xy[:, [0]])

        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

detabase_titanic = Titanic('train.csv')
train_data = DataLoader(dataset=detabase_titanic, batch_size=32, shuffle=True, num_workers=2)

# class net(torch.nn.Module):
#     def __init__(self):
#         super(net, self).__init__()
#         self.linear1 = torch.nn.Linear(6, 4)
#         self.linear2 = torch.nn.Linear(4, 2)
#         self.linear3 = torch.nn.Linear(2, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.activate = torch.nn.RReLU()
#
#     def forward(self, x):
#         x = self.activate(self.linear1(x))
#         x = self.activate(self.linear2(x))
#         x = self.sigmoid(self.linear3(x))
#         return x

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linear1 = torch.nn.Linear(6, 4)
        self.linear2 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        return x


model = net()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_draw = []

# TODO:criterion报错，说输入参数不在[0, 1]之间，但是debug显示y_pred和y都是满足要求的，未解决

if __name__ == '__main__':
    for epoch in range(100):
        for data in train_data:           # mini-batch的时间复杂度要比batch高很多
            x, y = data

            y_pred = model(x)
            loss = criterion(y_pred, y)
            # print('loss: %f' % (loss))
            loss_draw.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    plt.plot(loss_draw)
    plt.show()