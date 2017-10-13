import numpy as np
import layers as ly
import os

class NET():
    def __init__(self, input_shape, learning_rate=0.001):
        BS = input_shape[0]
        self.lr = learning_rate
        #conv1 : 1*28*28->6*12*12
        self.conv1 = ly.conv2d(input_shape, [5,5,1,6], [1,1],'same')
        self.conv1_relu = ly.relu()
        self.pool1 = ly.max_pooling(self.conv1.out_shape,[3,3],[2,2],'valid')
        # conv2 : 6*12*12 - > 10*5*5
        self.conv2 = ly.conv2d(self.pool1.out_shape, [3, 3, 6, 10], [1, 1], 'same')
        self.conv2_relu = ly.relu()
        self.pool2 = ly.max_pooling(self.conv2.out_shape, [3, 3], [2, 2],'valid')

        self.conv_fc = ly.conv_fc()
        self.fc1 = ly.full_connect(360,84)
        self.fc1_relu = ly.relu()
        self.fc2 = ly.full_connect(84, 10)
        self.loss = ly.softmax_cross_with_entropy()

    def forward(self, input, target=None):
        conv1 = self.conv1.forward(input)
        conv1_relu = self.conv1_relu.forward(conv1)
        pool1 = self.pool1.forward(conv1_relu)

        conv2 = self.conv2.forward(pool1)
        conv2_relu = self.conv2_relu.forward(conv2)
        pool2 = self.pool2.forward(conv2_relu)

        pool2_flatten = self.conv_fc.flatten(pool2)
        fc1 = self.fc1.forward(pool2_flatten)
        fc1_relu = self.fc1_relu.forward(fc1)
        self.fc2_out = self.fc2.forward(fc1_relu)
        #print('fc2',fc2)
        if target is not None:
            loss, pred = self.loss.forward(self.fc2_out, target)
            return loss, pred
        else:
            return self.loss.forward(self.fc2_out)

    def backward(self):
        dout = self.loss.backward()
        #print('d_fc2', dout)
        fc2_dout = self.fc2.backward(dout)
        fc1_relu_dout = self.fc1_relu.backward(fc2_dout)
        fc1_dout = self.fc1.backward(fc1_relu_dout)
        conv2_relu_unflatten_dout = self.conv_fc.unflatten(fc1_dout)

        pool2_dout = self.pool2.backward(conv2_relu_unflatten_dout)
        conv2_relu_dout = self.conv2_relu.backward(pool2_dout)
        conv2_dout = self.conv2.backward(conv2_relu_dout)

        poo1_dout = self.pool1.backward(conv2_dout)
        conv1_relu_dout = self.conv1_relu.backward(poo1_dout)
        conv1_dout = self.conv1.backward(conv1_relu_dout)

    def update(self):
        self.conv1.update(self.lr)
        self.conv2.update(self.lr)

        self.fc1.update(self.lr)
        self.fc2.update(self.lr)


    def save_model(self, filename):
        if not os.path.exists(filename):  ###判断文件是否存在，返回布尔值
            os.makedirs(filename)
        para_dict = {'conv1': [self.conv1.w_col, self.conv1.b],
                     'conv2': [self.conv2.w_col, self.conv2.b],
                     'fc1': [self.fc1.W, self.fc1.b],
                     'fc2': [self.fc2.W, self.fc2.b]}
        np.save(filename+'model.npy', para_dict)

    def load_model(self, filename):
        if not os.path.exists(filename):
            print('model_dir has not exist')
        param_dict = np.load(filename+'model.npy', encoding="latin1").item()
        self.conv1.w_col = param_dict['conv1'][0]
        self.conv1.b = param_dict['conv1'][1]

        self.conv2.w_col = param_dict['conv2'][0]
        self.conv2.b = param_dict['conv2'][1]

        self.fc1.W = param_dict['fc1'][0]
        self.fc1.b = param_dict['fc1'][1]

        self.fc2.W = param_dict['fc2'][0]
        self.fc2.b = param_dict['fc2'][1]
        print('load model ok')


