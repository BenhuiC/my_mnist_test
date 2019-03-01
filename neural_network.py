import os
import struct
from datetime import datetime

from scipy import special
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:

    def __init__(self, innodes, hnodes, onodes, lrate):
        self.inNodes = innodes
        self.hidNodes = hnodes
        self.outNodes = onodes
        self.learn_rate = lrate
        self.weight_of_in2hid = np.random.normal(0.0, pow(self.hidNodes, -0.5), (self.hidNodes, self.inNodes))
        self.weight_of_hid2out = np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hidNodes))
        self.active_func = lambda x: special.expit(x)

    def train(self, input_list, target_list):
        input = np.array(input_list, ndmin=2).T
        target = np.array(target_list, ndmin=2).T

        hidin = np.dot(self.weight_of_in2hid, input)
        hidout = self.active_func(hidin)

        outin = np.dot(self.weight_of_hid2out, hidout)
        out = self.active_func(outin)

        out_err = target - out

        hid_err = np.dot(self.weight_of_hid2out.T, out_err)
        self.weight_of_hid2out += self.learn_rate * np.dot((out_err * out * (1 - out)), np.transpose(hidout))

        self.weight_of_in2hid += self.learn_rate * np.dot((hid_err * hidout * (1 - hidout)), np.transpose(input))

    def query(self, input_list):
        input = np.array(input_list, ndmin=2).T

        hidin = np.dot(self.weight_of_in2hid, input)

        hidout = self.active_func(hidin)

        outin = np.dot(self.weight_of_hid2out, hidout)

        out = self.active_func(outin)
        return out


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

start_time=datetime.now()

input_nodes = 784
hiden_nodes = 200
out_nodes = 10
learn_rate = 0.2
n = NeuralNetwork(input_nodes, hiden_nodes, out_nodes, learn_rate)

# data_file = open('mnist_train_100.csv', 'r')
# data_list = data_file.readlines()
# data_file.close()

train_images, train_lable = load_mnist('/worktool/python/Neural_Network/mnist')

i = 0

while i < len(train_lable):
    img = train_images[i]
    target_value = train_lable[i]
    input = (np.asarray(img) / 255.0 * 0.99) + 0.01
    target = np.zeros(10) + 0.01
    target[target_value] = 0.99
    n.train(input, target)
    i += 1

end_time=datetime.now()
print("train take %d Seconds" % (end_time-start_time).total_seconds())
# test_file = open('mnist_test_10.csv', 'r')
# test_list = test_file.readlines()
# test_file.close()

test_images, test_lable = load_mnist('/worktool/python/Neural_Network/mnist', kind='t10k')

score = []
j = 0
while j < len(test_lable):
    img = test_images[j]
    target = test_lable[j]
    inputvalue = (np.asarray(img) / 255.0 * 0.99) + 0.01
    res = n.query(inputvalue)
    an = np.argmax(res)
    if an == target:
        score.append(1)
    else:
        score.append(0)
    j += 1

ss = np.asarray(score)
print('score is ', ss.sum() / ss.size)

'''
all = test_list[0].split(',')

data_in = (np.asfarray(all[1:]) / 255.0 * 0.99) + 0.01
result = n.query(data_in)

pri = np.asfarray(all[1:]).reshape((28, 28))

plt.title("image")
plt.imshow(pri, cmap="Greys", interpolation='None')
plt.show()
'''
