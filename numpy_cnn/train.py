import numpy as np
import net
import mnist
import sys

# if len(sys.argv) < 8:
#     print('Input: python train.py data_path model_path episode save_step val_step display_step batch_size learning_rate')
#     sys.exit()
#
# DATA_FILENAME = sys.argv[1]
# MODEL_FILENAME = sys.argv[2]
#
# EPISODE = int(sys.argv[3])
# SAVE_STEP = int(sys.argv[4])
# VAL_STEP = int(sys.argv[5])
# DISPLAY_STEP = int(sys.argv[6])
#
# BS = int(sys.argv[7])
# LR = float(sys.argv[8])

DATA_FILENAME = 'data/'
MODEL_FILENAME = 'model/'

EPISODE = 10000
SAVE_STEP = 5000
VAL_STEP = 100
DISPLAY_STEP = 20

BS = 32
LR = 0.0005


def compute_accuracy(predict, labels):
    error = [a - b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(labels, axis=-1))]
    correct = list(filter(lambda x: x == 0, error))
    accuracy = len(correct) / predict.shape[0]
    return accuracy

def train():
    for i in range(EPISODE):
        train_img,train_label = mni.train.next_batch(BS)
        train_img = train_img.reshape((BS, 1, 28, 28))
        train_loss,train_predict = mnist_net.forward(train_img, train_label)
        mnist_net.backward()
        mnist_net.update()
        #print('loss:', loss)
        if (i+1) % DISPLAY_STEP == 0:
            print(i+1, 'train_loss:', train_loss)
        if (i+1)% VAL_STEP == 0:
            print(i+1, '-------------------------------train_accuracy:', compute_accuracy(train_predict, train_label))
            val_img, val_label = mni.val.next_batch(5000)
            val_img = val_img.reshape((5000, 1, 28, 28))
            val_loss, val_predict = mnist_net.forward(val_img, val_label)
            print(i+1,'-------------------------------val_loss:', val_loss)
            print(i+1,'-------------------------------val_accuracy:', compute_accuracy(val_predict, val_label))
        if (i+1) % SAVE_STEP == 0:
            mnist_net.save_model(MODEL_FILENAME)
            print('save_model_',i+1)

if __name__ == '__main__':
    mnist_net = net.NET(learning_rate=LR, input_shape=[BS, 1, 28, 28])
    mni = mnist.read_data_sets(DATA_FILENAME, one_hot=True, reshape=False)
    train()