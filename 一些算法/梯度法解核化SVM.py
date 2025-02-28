import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
# 加载Iris数据集
iris = datasets.load_iris()
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

def kernelized_svm_gradient_descent(X, y, lambda_val, learning_rate, max_iter, kernel_function, gamma):
    """
    使用梯度下降法实现核化SVM。
    :param X: 特征矩阵
    :param y: 目标向量
    :param lambda_val: 正则化参数
    :param learning_rate: 学习率
    :param max_iter: 最大迭代次数
    :param kernel_function: 核函数
    :param gamma: 核函数参数
    :return: 模型参数
    """
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples)  # 初始化参数

    # 计算核矩阵
    K = kernel_function(X, X, gamma)

    for i in range(max_iter):
        # 计算梯度
        gradient = np.sum(K * (alpha * y[:, np.newaxis] * y), axis=1) - y
        # 添加正则化项
        gradient += lambda_val * alpha
        # 更新alpha
        alpha -= learning_rate * gradient

    return alpha

# 定义一个核函数（这里使用径向基函数）
def rbf_kernel_func(X1, X2, gamma):
    return np.exp(-gamma * np.sum((X1[:, np.newaxis] - X2)**2, axis=2))
# 使用梯度下降法训练模型
alpha_gd = kernelized_svm_gradient_descent(X_train, y_train, lambda_val=1.0, learning_rate=0.01, max_iter=1000, kernel_function=rbf_kernel_func, gamma=1.0)


# 使用scikit-learn的SVM包进行训练和预测
svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# 计算准确率
svm_accuracy = np.mean(svm_predictions == y_test)

print(svm_accuracy)


def predict_labels(alpha, X_test, y_test, X_train, kernel_function, gamma):
    """
    使用梯度下降法训练得到的alpha值来预测新数据点的类别。
    :param alpha: 模型参数
    :param X_test: 测试数据集
    :param y_test: 测试目标向量
    :param X_train: 训练数据集
    :param kernel_function: 核函数
    :param gamma: 核函数参数
    :return: 预测结果
    """
    n_samples, n_features = X_test.shape
    predictions = np.zeros(n_samples)

    # 计算核矩阵
    K_test = kernel_function(X_test, X_train, gamma)
    K_train = kernel_function(X_train, X_train, gamma)

    # 计算决策边界
    b = -np.sum(alpha * y_train[:, np.newaxis] * K_train) / np.sum(alpha * alpha)

    # 预测新数据点
    for i in range(n_samples):
        predictions[i] = np.sign(np.sum(alpha * y_train[:, np.newaxis] * K_test[i]) + b)

    return predictions

# 使用梯度下降法训练得到的alpha值来预测测试数据集
predictions_gd = predict_labels(alpha_gd, X_test, y_test, X_train, rbf_kernel_func, 1.0)

# 计算准确率
accuracy_gd = np.mean(predictions_gd == y_test)
print(accuracy_gd)

