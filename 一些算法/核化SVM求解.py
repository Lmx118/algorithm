import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将标签转换为{-1, 1}
y[y == 0] = -1

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义核函数
def rbf_kernel(X, Y, gamma):
    return np.exp(-gamma * np.sum((X[:, np.newaxis] - Y)**2, axis=2))

# 定义梯度下降法
def gradient_descent(X, y, learning_rate, lambda_reg, n_iterations):
    n_samples = X.shape[0]
    # 初始化偏置
    b = 0
    alphas = np.zeros(n_samples)

    for iteration in range(n_iterations):
        # 计算损失和梯度
        kernel_matrix = rbf_kernel(X, X)
        margins = y * (np.dot(kernel_matrix, alphas) + b)
        losses = np.maximum(0, 1 - margins)
        gradient_alphas = -y * losses + lambda_reg * alphas
        gradient_b = -y * np.sum(losses > 0)

        # 更新alpha和偏置
        alphas -= learning_rate * gradient_alphas
        b -= learning_rate * gradient_b

    return alphas, b

# 定义ADMM算法
def admm(X, y, lambda_reg, n_iterations, rho):
    n_samples = X.shape[0]
    # 初始化变量
    alphas = np.zeros(n_samples)
    z = np.zeros(n_samples)
    u = np.zeros(n_samples)
    b=0
    for iteration in range(n_iterations):
        # 更新alphas
        kernel_matrix = rbf_kernel(X, X)
        margins = y * (np.dot(kernel_matrix, alphas) + b)
        losses = np.maximum(0, 1 - margins)
        gradient_alphas = -y * losses + lambda_reg * alphas + rho * (z - u)
        alphas -= learning_rate * gradient_alphas

        # 更新z
        z = np.linalg.inv(np.eye(n_samples) * (rho + lambda_reg)) @ (rho * alphas + u)

        # 更新u
        u += alphas - z
# 计算b的梯度
        gradient_b = -np.sum(losses > 0)

        # 更新b
        b -= learning_rate * gradient_b

    return alphas, z, b


