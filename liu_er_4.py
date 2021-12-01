'''

这里是关于利用pytoch进行线性回归的问题，但是更新部分自己写的关于梯度的方法

'''

import numpy as np
import torch

ratio = 0.001

input = [1.0, 2.0, 3.0]
score = [2.0, 4.0, 6.0]

w_1 = torch.tensor([1.0], requires_grad = True)
w_2 = torch.tensor([3.0], requires_grad = True)
b = torch.tensor([1.0], requires_grad = True)

def forward(input):      # 二次模型不能很准确的得到值，说明模型的选择很重要
    return w_1*input**2 + w_2 * input + b

def loss(input, score):
    score_pre = forward(input)
    return (score_pre - score)**2


print('before training, I study %d hours a week, and i will get score: %d ' % (4.0, forward(4.0)))
for epoch in np.arange(5000):
    for input_x, score_y in zip(input, score):

        l = loss(input_x, score_y)

        l.backward()
        print('\tgrad:', w_1.grad.item(), w_2.grad.item(), b.grad.item())    # 使用.grad.item()输出该变量的值

        w_1.data = w_1.data - ratio*w_1.grad.data       # 这里手动进行权重更新
        w_2.data = w_2.data - ratio*w_2.grad.data
        b.data = b.data - ratio*b.grad.data

        w_1.grad.data.zero_()
        w_2.grad.data.zero_()
        b.grad.data.zero_()

    print('process: ', epoch, l.item())
print('after training, I exactlly get score: %f' % (forward(10.0).item()))
print('w_1 = %f, w_2 = %f, b = %f' % (w_1.item(), w_2.item(), b.item()))    # 注意用%d 和 %f 输出整数和浮点数


