import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DiabetesDataset(Dataset):            # 这个类需要返回数据的size，并且能够通过index访问内容
    def __init__(self, filepath):          # 需要用到的变量都在这里构建
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



database = DiabetesDataset('diabetes.csv.gz')        # 在这里犯错：变量名重复了
train_data = DataLoader(dataset=database, shuffle=True, batch_size=32, num_workers=2)


class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.linear1 = torch.nn.Linear(8, 4)
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

if __name__ == '__main__':
    for epoch in range(100):
        for data in train_data:           # mini-batch的时间复杂度要比batch高很多
            x, y = data

            y_pred = model(x)
            loss = criterion(y_pred, y)
            print('loss: %f' % (loss))
            loss_draw.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    plt.plot(loss_draw)
    plt.show()

