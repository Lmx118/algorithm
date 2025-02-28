import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class KernelizedSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='rbf', sigma=1.0):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.sigma = sigma
        self.alphas = None
        self.b = 0

    def rbf_kernel(self, X1, X2):
        # 计算RBF核
        gamma = 1.0 / (2 * self.sigma**2)
        dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * dists)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1) * 1.

        # 初始化alpha
        self.alphas = np.zeros((n_samples, 1))

        # 梯度下降
        for _ in range(self.n_iters):
            # 计算梯度
            kernel_matrix = self.rbf_kernel(X, X)
            y_pred = np.dot(kernel_matrix, self.alphas) + self.b
            errors = y_pred - y
            grad_alphas = np.sum(kernel_matrix * (errors * y), axis=1).reshape(-1, 1) + self.lambda_param * self.alphas
            grad_b = np.sum(errors * y)

            # 更新参数
            self.alphas -= self.lr * grad_alphas
            self.b -= self.lr * grad_b

    def predict(self, X):
        kernel_matrix = self.rbf_kernel(X, X_train)
        y_pred = np.dot(kernel_matrix, self.alphas) + self.b
        return np.sign(y_pred)

# 加载iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 只使用前两个特征，为了可视化
X = X[:, :2]

# 将标签转换为{-1, 1}
y[y == 0] = -1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建核化SVM模型并训练
svm = KernelizedSVM(learning_rate=0.01, lambda_param=0.01, n_iters=10000, kernel='rbf', sigma=1.0)
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)



class KernelizedSVMWithGradientDescent:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, sigma=1.0):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.sigma = sigma
        self.alphas = None
        self.b = 0

    def rbf_kernel(self, X1, X2):
        # 计算RBF核
        gamma = 1.0 / (2 * self.sigma**2)
        dists = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * dists)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        # 初始化alpha
        self.alphas = np.zeros((n_samples, 1))
        self.b = 0

        # 梯度下降
        for _ in range(self.n_iters):
            # 计算梯度
            kernel_matrix = self.rbf_kernel(X, X)
            y_pred = np.dot(kernel_matrix, self.alphas) + self.b
            errors = y_pred - y
            grad_alphas = np.sum(kernel_matrix * (errors * y), axis=1).reshape(-1, 1) + self.lambda_param * self.alphas
            grad_b = np.sum(errors * y)

            # 更新参数
            self.alphas -= self.lr * grad_alphas
            self.b -= self.lr * grad_b

    def predict(self, X):
        kernel_matrix = self.rbf_kernel(X, X_train)
        y_pred = np.dot(kernel_matrix, self.alphas) + self.b
        return np.sign(y_pred)

# 加载iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 只使用前两个特征，为了可视化
X = X[:, :2]

# 将标签转换为{-1, 1}
y[y == 0] = -1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建核化SVM模型并训练
svm = KernelizedSVMWithGradientDescent(learning_rate=0.001, lambda_param=0.01, n_iters=10000, sigma=1.0)
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(accuracy)


# # 计算准确率
# accuracy = np.mean(y_pred == y_test)
# print("Accuracy:", accuracy)
