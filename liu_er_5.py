import numpy as np
import torch
import matplotlib.pyplot as plt


class LinearModel (torch.nn.Module):    # 这是一个线性层，表示y=w*x+b
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pre = self.linear(x)
        return y_pre

class SquareModel_layer (torch.nn.Module):    # 这是一个二次型层，表示y=w*x**2+b，用两个线性层组合出来的
    def __init__(self):
        super(SquareModel_layer, self).__init__()
        self.layer_1 = torch.nn.Linear(1, 1, bias=False)
        self.layer_2 = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        y_1 = self.layer_1(x) * self.layer_1(x)
        y_2 = self.layer_2(y_1)
        return y_2



model = LinearModel()
print(model)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

criterion = torch.nn.MSELoss(size_average=False)

# 分别选择多个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.001)

loss_draw = []

for epoch in range(1000):
    y_pre = model(x)

    loss = criterion(y_pre, y)
    print('Epoch: %d, loss is : %f' % (epoch, loss))
    loss_draw.append(loss.item())   # loss 是一个tensor

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for parameters in model.parameters():
    print(parameters)

x_test = torch.tensor([[5.0]])
y_test = model(x_test)  # 测试时，输入model的数据仍然要与最初时一致（需要是tensor）
print('the finnal results about 4 is ', y_test.item())    # y_test.item()输出标量数字，y_test.data输出tensor


plt.plot(loss_draw)
plt.title('SGD')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()


