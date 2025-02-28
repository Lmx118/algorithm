import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time

# 生成数据集
X, y = make_regression(n_samples=10000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 核估计参数
bandwidth = 0.2  # 带宽参数，可以根据数据集调整

# 核估计函数，选择代表性的数据点
def kernel_density(X, y, bandwidth):
    """使用核密度估计选择代表性数据点"""
    # 这里简化处理，选择距离原点最近的一定数量的数据点作为代表性数据点
    # 实际应用中，可以根据核密度进行更复杂的选择
    distances = np.linalg.norm(X, axis=1)
    indices = np.argsort(distances)[:int(len(X) * bandwidth)]  # 选择前bandwidth%的数据点
    return X[indices], y[indices]

# 核估计选择数据点
X_train_kernel, y_train_kernel = kernel_density(X_train, y_train, bandwidth)

# 记录时间代价
start_time = time.time()

# 使用核估计选择的数据点训练初始模型
model_init = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_init.fit(X_train_kernel, y_train_kernel)

# 记录核估计和模型训练时间
kernel_and_training_time = time.time() - start_time

# 应用boosting
start_time = time.time()
model_boosted = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_boosted.fit(X_train, y_train)

# 记录boosting时间
boosting_time = time.time() - start_time

# 打印结果
print(f"Bandwidth: {bandwidth}")
print(f"Number of data points used after kernel density: {len(X_train_kernel)}")
print(f"Time taken for kernel density and initial model training: {kernel_and_training_time} seconds")
print(f"Time taken for boosting: {boosting_time} seconds")

# 评估模型性能
y_pred_init = model_init.predict(X_test)
y_pred_boosted = model_boosted.predict(X_test)
mse_init = mean_squared_error(y_test, y_pred_init)
mse_boosted = mean_squared_error(y_test, y_pred_boosted)

print(f"MSE for initial model: {mse_init}")
print(f"MSE for boosted model: {mse_boosted}")