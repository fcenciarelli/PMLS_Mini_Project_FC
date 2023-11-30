import numpy as np

# inspired from https://github.com/zhangjh915/Simple-Convolutional-Neural-Network/blob/master/layers.py
# inspired from https://github.com/vcoz17/Back-Propagation-in-CNN/blob/master/backpropagation_in_cnn.py



def conv_forward(x, w, b):

  N, C, H, W = x.shape 
  F, _, FH, FW = w.shape

  stride = 1                  
  padding =  (FH - 1) // 2   

  out = np.zeros((N, F, H, W))

  padded_x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
  _, _, padded_H, padded_W = padded_x.shape

  x_col = np.zeros((C * FH * FW, H * W))
  w_row = w.reshape(F, C * FH * FW)

  for i in range(N):
    c = 0
    for j in range(0, padded_H - FH + 1, stride):
      for k in range(0, padded_W - FW + 1, stride):
        x_col[:, c] = padded_x[i, :, j:j+FH, k:k+FW].reshape(C * FH * FW)
        c += 1
    out[i, :] = (np.dot(w_row, x_col) + b.reshape(-1, 1)).reshape(F, H, W)
  
  cache = (x, w, b, stride, padding)
  return out, cache 


def max_pool_forward(x, shape=[2, 2], stride=2):

    N, C, H, W = x.shape
    pool_heigt, pool_width = shape
    out_H = 1 + (H - pool_height) // stride
    out_W = 1 + (W-pool_width) // stride

    out = np.zeros((N, C, out_H, out_W))

    for i in range(N):
        curr_out  = np.zeros((C, out_H* out_W))
        c = 0
        for j in range(0, H - pool_height + 1, stride):
            for k in range(0, W - pool_width + 1, stride):
                curr_region = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height * pool_width)
                curr_max_pool = np.max(curr_region, axis=1)
                curr_out[:, c] = curr_max_pool
                c += 1
        out[i, :, :, :] = curr_out.reshape(C, out_H, out_W)

    cache = (x, pool_height, pool_width, stride)
    return out, cache


def max_pool_backward(dout, cache):
    x, pool_height, pool_width, stride = cache 
    N, C, H, W = x.shape
    _, _, out_H, out_W = dout.shape 

    dx = np.zeros_like(x)

    for i in range(N):
        curr_dout = dout[i, :].reshape(C, out_H * out_W)
        c = 0
        for j in range(0, H - pool_height + 1, stride):
            for k in range(0, W - pool_width + 1, stride):
                curr_region = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height * pool_width)
                curr_max_idx = np.argmax(curr_region, axis=1)
                curr_dout_region = curr_dout[:, c]
                curr_dpooling = np.zeros_like(curr_region)
                curr_dpooling[np.arange(C), curr_max_idx] = curr_dout_region
                dx[i, :, j:j+pool_height, k:k+pool_height] = curr_dpooling.reshape(C, pool_height, pool_width)
                c += 1
    
    return dx 


def relu_forward(x):

    out = np.maximum(x, 0 )
    cache = x

    return out, cache


def relu_backward(dout, cache):

    x = cache
    dx = dout * (x > 0)

    return dx



def FC_forward(x, w, b):

    backprop_cache = (x, w, b)
    x = x.reshape(x.shape[0], -1)
    out = x.dot(w) + b
    return out, backprop_cache


def FC_backward(dout, cache):

    x, w, b = cache

    x_new = x.reshape(x.shape[0], -1)
    
    dx = np.dot(dout, w.T).reshape(x.shape)
    dw = np.dot(x_new.T, dout)
    db = np.sum(dout.T, axis =1)


    return dx, dw , db


def softmax_loss(x, y):
    
    x = x - np.max(x, axis=1, keepdims = True)
    numerator = np.exp(x)
    probs = numerator / np.sum(numerator, axis= 1, keepdims = True)

    loss = -np.sum(np.log(probs[np.arange(x.shape[0]), y])) / x.shape[0]

    dx = probs.copy()
    dx[np.arange(x.shape[0]), y] -= 1
    dx /= x.shape[0]

    return loss, dx







#Testing FC forward
def testing_FC_forward():
    n = 10
    m = 5
    p = 3
    x = np.arange(n * m).reshape(n, m)
    w = np.arange(m * p).reshape(m, p)
    b = np.arange(p)

    out, cache = FC_forward(x, w, b)

    print(out)

def testing_FC_backward():
    n, m, p = 2, 3, 2

    x = np.arange(1, n * m + 1).reshape(n, m)
    w = np.arange(n * m + 1, n * m + m * p + 1).reshape(m, p)
    dout = np.arange(n * m + m * p + 1, n * m + m * p + n * p + 1).reshape(n, p)

    cache = (x, w, np.zeros(p))

    dx, dw, db = FC_backward(dout, cache)

    print("dx:\n", dx)
    print("\ndw:\n", dw)
    print("\ndb:\n", db)



def testing_conv_forward():
    N, C, H, W = 1, 1, 4, 4  
    F, FH, FW = 1, 3, 3  

    x = np.arange(1, N * C * H * W + 1).reshape(N, C, H, W)
    w = np.array([[[[1,2,3], 
                    [1, 2,3],
                    [1, 2, 3]]]])
    b = np.zeros(F)

    out = conv_forward(x, w, b)

    print("Input:\n", x)
    print("\nFilter:\n", w)
    print("\nBias:\n", b)
    print("\nOutput:\n", out)



testing_conv_forward()


# class ConvNet(object):
#     def __init__(self, 
#                  input_dim = (1, 28, 28),
#                  hidden_dim = 64, 
#                  num_classes = 10,
#                  weight_scale=0.01,
#                  reg=0.0):
        
#         C, H, W = input_dim

#         # (64, C, 3, 3) 64 filters, C channels, 3x3 filter
#         # (64, ) bias size
#         self.W1 = np.random.normal(0.0, weight_scale, (64, C, 3, 3))
#         self.b1= np.zeros((64 ,))

#         # (64, 64, 3, 3), second 64 is number of input from the previous layer
#         self.W2 = np.random.normal(0.0, weight_scale, (64, 64, 3, 3))
#         self.b2 = np.zeros((64, ))

#         conv_out_H = 28 // 4 
#         conv_out_W = 28 // 4

#         self.W3 = np.random.randn(64 * conv_out_H * conv_out_W, hidden_dim) * np.sqrt(2.0 / (64 * conv_out_H * conv_out_W))
#         self.b3 = np.zeros((hidden_dim, ))
#         self.W4 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
#         self.b4 = np.zeros((num_classes, ))

#         self.reg = reg

#     def forward(self, x):

#         x, conv1_cache = conv
