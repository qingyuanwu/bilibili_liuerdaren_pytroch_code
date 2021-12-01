
'''

这里是关于成功或失败的二值分类问题

'''

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Binary_classfication(torch.nn.Module):
    def __init__(self):
        super(Binary_classfication, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pre = torch.sigmoid(self.linear(x))        # 这里原本是F.sigmoid()，但是更新后可以直接使用torch.sigmoid()
        return y_pre

model = Binary_classfication()

critizer = torch.nn.BCELoss(reduction='sum')         # 这里原本是size_average=False，更新后使用reduction='sum'
'''
reduction='sum'应该是没有平均，lr=0.001。reduction='mean'，lr=0.01才能达到前者的效果，体会lr和loss之间的关系
'''
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[0.5], [0.7], [0.9], [1.0], [1.2], [1.5], [1.7], [2.1], [2.2], [2.6], [3.1], [3.5], [4.2]])
y = torch.tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1]])

for epoch in range(10000):
    y_pos = model(x)
    y = y.to(torch.float32)             # 下一步critizer会将y认成long，这里先将其转换为float以满足要求
    loss = critizer(y_pos, y)
    print('Epoch: %d, loss: %f' % (epoch, loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, 200)
x_test = torch.tensor(x).view(len(x), -1)
x_test = x_test.to(torch.float32)      # 同上

y_test = model(x_test)
y = y_test.data.numpy()

plt.plot(x, y)
plt.show()

