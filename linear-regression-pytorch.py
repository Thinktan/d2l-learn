
import torch
from IPython import display
import numpy as np
import random
from d2lzh_pytorch import *

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs, dtype=torch.float32)

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

print(features[0], labels[0])

# 第二个特征features[:,1]与标签labels的散点图
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
#plt.show()


# 读取数据
import torch.utils.data as Data

batch_size = 10

# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

#for X, y in data_iter:
#    print(X, y)
#    break

# 定义模型
print("定义模型")
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)

# 初始化模型参数
print("初始化模型参数")
from torch.nn import init
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
print(net.linear.weight)
print(net.linear.bias)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
print("定义优化算法")
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 训练模型
print("训练模型")
num_epochs = 3
for epoch in range(1, num_epochs+1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print("epoch %d, loss:%f" % (epoch, l.item()))

# 输出结果
print("输出结果")

print(true_w, net.linear.weight)
print(true_b, net.linear.bias)




