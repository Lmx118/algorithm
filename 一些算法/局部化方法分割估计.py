import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import time

# 生成数据集
X, y = make_regression(n_samples=100000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 局部化方法 - 这里以简单的分割为例
n_splits = 5
for i in range(n_splits):
    start_time = time.time()
    # 随机分割数据
    X_train_part = X_train[i::n_splits]
    y_train_part = y_train[i::n_splits]
    num=len(X_train_part)
    # 训练模型
    model = AdaBoostRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_part, y_train_part)

    # 测试模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    end_time = time.time()

    print(f"Split {i + 1}, Train Number:{num}, MSE: {mse}, Time: {end_time - start_time} seconds")

# 记录参数对算法的影响
# 这里可以添加更多的参数调整和记录