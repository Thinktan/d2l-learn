

#from IPython import display
#from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from d2lzh import set_figsize
from d2lzh import plt
from d2lzh import data_iter


# 生成标签数据

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))

labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b

labels += nd.random.normal(scale=0.01, shape=labels.shape)

plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(features[:,1].asnumpy(), labels.asnumpy())
#plt.show();


# 读取数据集

batch_size = 10

for x, y in data_iter(batch_size, features, labels):
    print(x, y)
    break

# 初始化模型参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
print(w, b)

# 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 定义损失函数
# 预测值: y_hat
# 真实值：y
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

