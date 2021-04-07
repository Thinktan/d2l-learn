
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
batch_size = 10
#for X, y in data_iter(batch_size, features, labels):
#    print(X, y)
#    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 过调用反向函数backward计算小批量随机梯度，并调用优化算法sgd迭代模型参数。
lr = 0.03
num_epochs = 3
# 定义模型
#def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
#    return torch.mm(X, w) + b
net = linreg

# 定义损失函数
#def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
#    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
#    return (y_hat - y.view(y_hat.size())) ** 2 / 2
loss = squared_loss

# 定义优化算法sgd
#def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
#    for param in params:
#        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


# 训练模型
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w, b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print("----------")

print("w真实值：", true_w)
print("w训练值：", w)
print("----------")
print("b真实值：", true_b)
print("b训练值：", b)

