
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1)*0.01, 0.05

#print(true_w)
#print(true_b)

features = torch.randn((n_train+n_test, num_inputs))
print(features)
