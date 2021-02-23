

#from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))

labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b

labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(features.shape)
print(labels.shape)