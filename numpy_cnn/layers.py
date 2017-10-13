import numpy as np

from im2col import im2col, col2im,get_indices

# match with tensorflow , only forward, backward is not changed now
class conv2d():
    def __init__(self, input_shape, filter_shape, stride_shape, padding='same'):
        self.input_shape = input_shape
        self.BS, self.in_D, self.in_H, self.in_W = self.input_shape #[BS, in_D, in_H, in_W]
        self.f_H, self.f_W, _, self.out_D = filter_shape # filter_H, filter_W is odd
        self.s_H, self.s_W = stride_shape
        self.p_H, self.p_W = 0, 0
        if padding == 'same':
            out_H = int(int(self.in_H / self.s_H - 0.5) + 1)
            int_H_ = out_H * self.s_H
            self.p_H = int_H_ - self.s_H + self.f_H - self.in_H

            out_W = int(int(self.in_W / self.s_W - 0.5) + 1)
            int_W_ = out_W * self.s_W
            self.p_W = int_W_ - self.s_W + self.f_W - self.in_W

        self.out_H = int((self.in_H + self.p_H - self.f_H) / self.s_H + 1)
        self.out_W = int((self.in_W + self.p_W - self.f_W) / self.s_W + 1)

        self.out_shape = [self.BS, self.out_D, self.out_H, self.out_W]

        # HE Initialization : consider input only
        Weight = np.sqrt(2./(self.f_H * self.f_W * self.in_D)) * np.random.randn(self.out_D, self.in_D* self.f_H * self.f_W)
        self.w_col = Weight.reshape((self.out_D, -1))
        self.b = 0. * np.ones(shape=[self.out_D, 1])

    def forward(self, input):
        # in_col : [f_H*f_W*in_D, out_H*out_W*BS]
        self.in_col = im2col(input,[self.f_H, self.f_W], [self.s_H, self.s_W], [self.p_H, self.p_W])
        out_col = np.matmul(self.w_col, self.in_col) + self.b #[out_D, out_H*out_W*BS]
        out = out_col.reshape((self.out_D, self.out_H, self.out_W, input.shape[0])) #[out_D, out_H, out_W, BS]
        out = np.transpose(out,(3,0,1,2))  #[BS, out_D, out_H, out_W]
        return out

    def backward(self, dout):
        # dout : [BS, out_D, out_H, out_W]
        # self.in_col : [f_H*f_W*in_D, out_H*out_W*BS]
        self.db = np.sum(dout, axis=(0,2,3)).reshape((self.out_D,1)) # [out_D, 1]

        dout_reshape = np.transpose(dout, (1,2,3,0)).reshape((self.out_D,-1)) #[out_D, out_H *out_W * BS]
        self.dw = np.matmul(dout_reshape, self.in_col.T) #[out_D, in_D*f_H*f_W]

        din_col = np.matmul(self.w_col.T, dout_reshape) # [f_H*f_W*in_D, out_H *out_W * BS]
        din = col2im(din_col, self.input_shape, [self.f_H, self.f_W], [self.s_H, self.s_W], [self.p_H, self.p_W])
        return din #[BS, in_D, in_H, in_W]

    def update(self, lr, mode='SGD'):
        if mode == 'SGD':
            self.w_col -= lr * self.dw
            self.b -= lr * self.db

class max_pooling():
    def __init__(self,input_shape, filter_shape, stride_shape, padding='same'):
        self.input_shape = input_shape
        self.BS, self.in_D, self.in_H, self.in_W = input_shape  # [BS, in_D, in_H, in_W]
        self.f_H, self.f_W = filter_shape  # filter_H, filter_W is odd
        self.s_H, self.s_W = stride_shape
        self.p_H, self.p_W = 0, 0
        if padding == 'same':
            out_H = int(int(self.in_H / self.s_H - 0.5) + 1)
            int_H_ = out_H * self.s_H
            self.p_H = int_H_ - self.s_H + self.f_H - self.in_H

            out_W = int(int(self.in_W / self.s_W - 0.5) + 1)
            int_W_ = out_W * self.s_W
            self.p_W = int_W_ - self.s_W + self.f_W - self.in_W

        self.out_H = int((self.in_H + self.p_H - self.f_H) / self.s_H + 1)
        self.out_W = int((self.in_W + self.p_W - self.f_W) / self.s_W + 1)

        self.out_shape = [self.BS, self.in_D, self.out_H, self.out_W]

    def forward(self, input):
        # in_col : [f_H*f_W*in_D, out_H*out_W*BS]
        self.in_col = im2col(input, [self.f_H, self.f_W], [self.s_H, self.s_W], [self.p_H, self.p_W])
        in_col_reshape = self.in_col.reshape((self.f_H*self.f_W,-1),order='F') # [f_H*f_W, out_H*out_W*BS*in_D] ???????
        out_col = np.max(in_col_reshape, axis=0) # [1, out_H*out_W*BS*in_D]
        out = out_col.reshape((self.in_D, -1),order='F')
        out = out.reshape((self.in_D, self.out_H, self.out_W, input.shape[0])) # [in_D, out_H, out_W, BS]
        out = np.transpose(out, (3, 0, 1, 2))  # [BS, in_D, out_H, out_W]
        arg_index = np.argmax(in_col_reshape, axis=0) # [1, out_H*out_W*BS*in_D]
        self.W = np.zeros(in_col_reshape.shape)
        self.W[arg_index, np.arange(in_col_reshape.shape[1])] = 1. # [f_H*f_W, out_H*out_W*BS*in_D]

        return out

    def backward(self, dout):
        # dout : [BS, in_D, out_H, out_W]
        # self.in_col : [f_H*f_W*in_D, out_H*out_W*BS]
        dout_reshape = dout.transpose((2,3,0,1)).reshape((-1, self.in_D)).T.reshape((1,-1),order='F')  # [ out_H *out_W * BS*in_D]
        din_col = self.W * dout_reshape  # [f_H*f_W,  out_H *out_W * BS * in_D]
        din_col = din_col.reshape((self.f_H * self.f_W * self.in_D, -1), order='F') # [f_H*f_W * in_D, out_H *out_W * BS]
        din = col2im(din_col, self.input_shape, [self.f_H, self.f_W], [self.s_H, self.s_W], [self.p_H, self.p_W])
        return din  # [BS, in_D, in_H, in_W]

class full_connect():
    def __init__(self, in_len, out_len):
        self.out_len = out_len
        self.in_len = in_len
        self.W = np.sqrt(2./in_len) * np.random.randn(in_len, out_len)
        self.b = 0. * np.zeros((1, out_len))

    def forward(self, input):
        self.input = input
        return np.matmul(input, self.W) + self.b

    def backward(self, dout):
        # dout : [BS, out_len]
        # input [BS, in_len]
        self.db = np.sum(dout, axis=0).reshape(1,self.out_len)

        in_reshape = self.input.reshape((-1, self.in_len, 1))
        dout_reshape = dout.reshape((-1, 1, self.out_len))
        self.dw = np.sum(in_reshape * dout_reshape, axis=0)

        din = np.matmul(dout, self.W.T)
        return din

    def update(self, lr, mode='SGD'):
        if mode == 'SGD':
            self.W -= lr * self.dw
            self.b -= lr * self.db

class relu():
    def forward(self, input):
        self.output = np.maximum(input, 0) * 1.0
        return self.output

    def backward(self, dout):
        din =  np.sign(self.output) * dout
        return din

class sigmoid():
    def forward(self, input):
        self.out = 1.0 / (1.0 + np.exp(-input))
        return self.out

    def backward(self, dout):
        din = self.out * (1.0 - self.out) * dout

class tanh():
    def forward(self, input):
        in_ = np.maximum(np.minimum(input, 5e1), -5e1)
        exp = np.exp(in_)
        exp_ = np.exp(-in_)
        self.out = (exp - exp_) / (exp + exp_)
        return self.out

    def backward(self, dout):
        din = (1. - np.power(self.out, 2)) * dout
        return din

class softmax_cross_with_entropy():
    def forward(self, input, labels=None): # one_hot
        self.input = input
        self.labels = labels
        in_ = self.input / np.max(self.input, axis=-1)
        in_exp = np.exp(in_)
        self.prob = in_exp / np.sum(in_exp, axis=-1).reshape((-1,1))
        #print('prob', prob)
        # input<1e-10 -> input = 1e-10
        if labels is not None:
            self.prob = np.minimum(1, np.maximum(self.prob, 1e-10))
            sum = np.sum(labels*np.log(self.prob), axis=-1)
            loss = -np.mean( sum)
            return loss, self.prob
        else:
            return self.prob

    def backward(self):
        return self.prob - self.labels

class square_loss():
    def forward(self, input, target):
        self.input = input
        self.target = target
        loss = np.mean( np.sum((input-target)**2, axis=-1))
        return loss, input

    def backward(self):
        return 2 * (self.input - self.target)

class conv_fc():
    def flatten(self, input):
        self.input_shape = input.shape
        #[BS, D, H, W] -> [BS, H, W, D]
        return input.reshape((input.shape[0],-1)) #[BS, -1]
    def unflatten(self, dout):
        return dout.reshape(self.input_shape)