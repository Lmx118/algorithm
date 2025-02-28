import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import time

# k近邻的k值
k = 15

# 生成数据集
X, y = make_regression(n_samples=100000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=k)

# 记录时间代价
start_time = time.time()

# 第一步：选择最接近测试点的k个样本
knn_reg.fit(X_train, y_train)
distances, indices = knn_reg.kneighbors(X_test)

# 第二步：对局部样本进行平均处理
X_train_local = np.array([X_train[idx].mean(axis=0) for idx in indices])
y_train_local = np.array([y_train[idx].mean(axis=0) for idx in indices])

# 记录局部化方法的时间和采用的数据量
localization_time = time.time() - start_time
print(f"Localization time: {localization_time} seconds")
print(f"Number of localized data points: {len(X_train_local)}")

# 再用boosting算法处理
start_time = time.time()
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_local, y_train_local)

# 记录boosting算法的时间代价
boosting_time = time.time() - start_time
print(f"Boosting time: {boosting_time} seconds")

# 评估模型性能
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE for the model: {mse}")

# 记录最终采用的数据量对算法的影响
print(f"Final number of data points used: {len(X_train_local)}")