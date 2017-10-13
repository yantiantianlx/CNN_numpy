import numpy as np
import net
import mnist
import sys

# if len(sys.argv) < 4:
#     print('Input: python test.py data_path model_path test_number')
#     sys.exit()
#
# DATA_FILENAME = sys.argv[1]
# MODEL_FILENAME = sys.argv[2]
# BS = int(sys.argv[3]) #训练 batch_size
# if BS > 10000:
#     print('Error: test_number > 10000')
#     sys.exit()

DATA_FILENAME = 'data/'
MODEL_FILENAME = 'model/'
BS = 10000


def compute_accuracy(predict, labels):
    error = [a - b for a, b in zip(np.argmax(predict, axis=-1), np.argmax(labels, axis=-1))]
    correct = list(filter(lambda x: x == 0, error))
    accuracy = len(correct) / BS
    return accuracy


if __name__ == '__main__':
    mnist_net = net.NET(input_shape=[BS, 1, 28, 28])
    mnist_net.load_model(MODEL_FILENAME)
    mni = mnist.read_data_sets(DATA_FILENAME, one_hot=True,reshape=False)
    imgages, labels = mni.test.next_batch(BS)
    imgages = imgages.reshape((BS, 1, 28, 28))
    loss, pred = mnist_net.forward(imgages, labels)
    print('test_loss',loss)
    print('test_accuracy', compute_accuracy(pred, labels))

