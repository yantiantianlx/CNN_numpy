import numpy as np
import cv2
from net import NET
import sys
import os

if len(sys.argv) < 3:
    print('Input: python test_img.py model_path img_path')
    sys.exit()
BS = 1
img_path = sys.argv[1]
MODEL_FILENAME = sys.argv[2]
mnist_net = NET(input_shape=[BS, 1, 28, 28])
mnist_net.load_model(MODEL_FILENAME)


for file in os.listdir(img_path):
    img = cv2.imread(img_path+file)
    img = cv2.resize(img, (28,28))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_array = np.array(img, dtype=np.float32).reshape((1,1,28,28)) / 255.

    pred = mnist_net.forward(img_array)
    pred = np.argmax(pred[0])
    print(file, ': ', pred)