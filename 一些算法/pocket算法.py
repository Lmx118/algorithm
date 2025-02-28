import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import *
import random
import pandas as pd

# dataset = [[0.10 ,-0.10, 1],
#            [0.30 , 0.60, 1],
#            [0.50 ,-0.20, 1],
#            [0.60 ,-0.25, 1],
#            [-0.10,-0.25, 1],
#            [-0.42,-0.30, 1],
#            [-0.50,-0.15,-1],
#            [-0.55,-0.12,-1],
#            [-0.70,-0.28,-1],
#            [-0.51, 0.22,-1],
#            [-0.48, 0.48,-1],
#            [-0.52, 0.47,-1],
#            [ 0.15, 0.63,-1],
#            [ 0.09, 0.81,-1],
#            [-0.68, 0.58,-1]]


dataset = [[0.10, -0.10, 1],
           [0.00, 0.75, 1],
           [0.50, -0.20, 1],
           [0.60, -0.25, 1],
           [-0.10, -0.25, 1],
           [-0.55, 0.30, 1],
           [-0.50, -0.15, -1],
           [-0.55, -0.12, -1],
           [0.53, -0.28, -1],
           [-0.51, 0.22, -1],
           [-0.48, 0.48, -1],
           [-0.52, 0.47, -1],
           [0.15, 0.63, -1],
           [0.09, 0.81, -1],
           [-0.68, 0.58, -1]]

dataset = np.array(dataset)  # 变为数组，不然无法进行分片操作


def Display(dataset, w0, w1, b):  # 结果图
    plt.scatter(dataset[0:6, 0], dataset[0:6, 1], color='blue', marker='o', label='Positive')
    plt.scatter(dataset[6:, 0], dataset[6:, 1], color='red', marker='x', label='Negative')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.title('POCKET')
    plt.plot([-1, 1], [(w0 - b) / w1, -1 * (w0 + b) / w1], 'g')  # 画直线
    plt.show()


def num_fault(W, b, dataset):
    count = 0
    for i in range(0, len(dataset)):
        X = np.array(dataset[i][:-1])
        Y = np.dot(W, X) + b
        if sign(Y) == sign(dataset[i][-1]):
            continue
        else:
            count += 1
    return count


def POCKET():
    starttime = time.time()
    count = 0
    W = np.ones(2)
    b = 0
    best_W = W
    best_b = b
    min_num = 15
    while True:
        count += 1
        end = True
        faultset = []
        for i in range(0, len(dataset)):
            x = dataset[i][:-1]
            X = np.array(x)
            Y = np.dot(W, X) + b
            if sign(Y) == sign(dataset[i][-1]):
                continue
            else:
                end = False
                faultset.append(dataset[i])
        if end == False:
            j = random.randint(0, len(faultset) - 1)
            W = W + (faultset[j][-1]) * faultset[j][:-1]
            b = b + faultset[j][-1]
            num = num_fault(W, b, dataset)
            if num < min_num:
                count = 0
                min_num = num
                best_W = W
                best_b = b
        if (end or count == 100):  # 限制迭代次数上限为100次
            break
    endtime = time.time()
    dtime = endtime - starttime
    print("time: %.8s s" % dtime)
    print("best_W:", best_W)
    print("best_b:", best_b)
    print("count:", count)
    print("min_fault_point:", min_num)
    Display(dataset, best_W[0], best_W[1], best_b)
    return best_W


def main():
    W = POCKET()


if __name__ == '__main__':
    main()
