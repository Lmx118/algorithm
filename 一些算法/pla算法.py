import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import *
import random

dataset = [[0.10 ,-0.10, 1],
           [0.30 , 0.60, 1],
           [0.50 ,-0.20, 1],
           [0.60 ,-0.25, 1],
           [-0.10,-0.25, 1],
           [-0.42,-0.30, 1],
           [-0.50,-0.15,-1],
           [-0.55,-0.12,-1],
           [-0.70,-0.28,-1],
           [-0.51, 0.22,-1],
           [-0.48, 0.48,-1],
           [-0.52, 0.47,-1],
           [ 0.15, 0.63,-1],
           [ 0.09, 0.81,-1],
           [-0.68, 0.58,-1]]


dataset = np.array(dataset)  # 变为数组，不然无法进行分片操作
def Plot(dataset, w0, w1, b):  # 结果图
    plt.scatter(dataset[0:6, 0], dataset[0:6, 1], color='blue', marker='o', label='Positive')
    plt.scatter(dataset[6:, 0], dataset[6:, 1], color='red', marker='x', label='Negative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.title('PLA')
    plt.plot([-1, 1], [(w0 - b) / w1, -1 * (w0 + b) / w1], 'g')  # 画直线
    plt.show()

def PLA():
    W = np.zeros(2)
    b = 0
    num = 0
    starttime = time.time()
    while True:
        num += 1
        end = True
        for i in range(0, len(dataset)):
            x = dataset[i][:-1]
            X = np.array(x)
            Y = np.dot(W, X) + b
            if sign(Y) == sign(dataset[i][-1]):
                continue
            else:
                end = False
                W = W + (dataset[i][-1]) * X
                b = b + dataset[i][-1]

        if end == True:
            break
    endtime = time.time()
    dtime = endtime - starttime
    print("W:", W)
    print("count:", num)
    print("time: %.8s s" % dtime)
    Plot(dataset, W[0], W[1], b)
    return W


def main():
    W = PLA()


if __name__ == '__main__':
    main()