import numpy as np


def FC_forward(x, w, b):

    backprop_cache = (x, w, b)
    x = x.reshape(x.shape[0], -1)
    out = x.dot(w) + b
    return out, backprop_cache



x = np.random.randn(10, 5)
w = np.random.randn(5, 3)
b = np.random.randn(3)

out, cache = FC_forward(x, w, b)

print(out)