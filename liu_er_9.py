import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
'''
这是关于线性层进行MNIST集分类的程序
'''
batch_size = 64                   # 设置mini-batch

transform = transforms.Compose([
    transforms.ToTensor(),                            # 将数据transform为tensor
    transforms.Normalize((0.1307, ), (0.3081, ))      # 将数据进行normalize操作，将图片的值域限制在[0, 1]之间
])

train_data = datasets.MNIST(root='dataset/mnist', train=True, download=True, transform=transform)  # 加载训练集，利用torchvision加载

train_load = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.MNIST(root='dataset/mnist', train=False, download=True, transform=transform)  # 加载测试集，利用torchvision加载

test_load = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)         # 控制变量法，测试集不用打乱顺序

class net(torch.nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)         # 要用softmax()损失函数，所以最后一层不加激活函数
        return x

model = net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum可以加速同一方向的优化进度

def train(epoch):
    loss_draw = []
    for k in np.arange(epoch):

        for batch_index, data in enumerate(train_load):
            inputs, traget = data
            img_pred = model(inputs)

            loss = criterion(img_pred, traget)
            loss_draw.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 300 == 299:
                print('batch_index is %d, loss is %f' % (batch_index, loss.item()))

def test():
    with torch.no_grad():
        print('predict              traget')
        corrct = 0.0
        total = 0.0
        for data in test_load:
            inputs, traget = data

            outputs = model(inputs)
            _, img_pred = torch.max(outputs.data, dim=1)        # dim=1表示列
            corrct += (img_pred == traget).sum().item()
            total += traget.size(0)

        for k in np.arange(traget.size(0)):            # 输出最后一组mini-batch的情况
            print('%d                   %d' % (img_pred[k].item(), traget[k].item()))
        print('total_tset_corrct_ratio is %f' % (corrct/total))


if __name__ == '__main__':

    epoch = 1
    train(epoch)

    test()


