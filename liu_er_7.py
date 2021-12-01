
'''

这是关于多维输入的分类问题，8个输入特征，1个输出

'''

import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)   # delimiter：以逗号为分解，一般1080，2080显卡都用np.float32
x = torch.from_numpy(xy[:750, :-1])
y = torch.from_numpy((xy[:750, [-1]]))     # 虽然y只取一列，但仍然要保证它与x一致，都是二维矩阵

x_test = torch.from_numpy(xy[750:, :-1])    # 取出最后9个样本作为测试集
y_test = torch.from_numpy(xy[750:, [-1]])

class Multilayer(torch.nn.Module):
    def __init__(self):
        super(Multilayer, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmiod = torch.nn.Sigmoid()      # 为了保证最后一层输出是平滑的，所以最后一层始终采用sigmoid()

        # self.activate = torch.nn.Sigmoid()   # 尝试多个激活函数，看看各自效果
        # self.activate = torch.nn.ReLU()      # 根据不同的激活函数，灵活地调整lr
        # self.activate = torch.nn.Tanh()
        self.activate = torch.nn.LogSigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.sigmiod(self.linear4(x))

        return x

model = Multilayer()

critzier = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

loss_draw = []

for epoch in range(10000):     # 增加训练次数模型会好很多
    y_pre = model(x)

    loss = critzier(y_pre, y)
    print('epoch: %d, loss: %f' % (epoch, loss))
    loss_draw.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_pred = model(x_test)

y_pred = y_pred.data.numpy()       # 输出时，尽量都转换成numpy()
y_test = y_test.data.numpy()

print('prediction         label ')
for k in range(9):
    print(y_pred[k],'      ',  y_test[k])

plt.plot(loss_draw)    # 绘制loss曲线，不管弹窗会不运行，所以放在最后
plt.show()

