
import torch
import torchvision
import numpy as np
import sys
sys.path.append(".") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

print("获取和读取数据")
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

print("初始化模型参数")
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

print("实现softmax运算")
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
#print(X_prob, X_prob.sum(dim=1))

print("定义模型")
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
