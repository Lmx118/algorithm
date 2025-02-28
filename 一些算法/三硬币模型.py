import numpy as np
# 硬币 A、B、C 正面向上的概率
pi = 0.7
p = 0.3
q = 0.6
# 进行 100 次试验
n = 100
# 抛硬币 A 的结果
coin_A = np.random.rand(n) < pi
# 根据硬币 A 的结果选择 B 或 C 并抛掷
coin_B_or_C = np.where(coin_A, np.random.rand(n) < p, np.random.rand(n) < q)
# 记录观测结果序列
observations = np.where(coin_A, coin_B_or_C, 0)  # 如果硬币 A 正面则记录硬币 B 或 C 的结果，否则记录为 0
# 打印观测结果序列
print("观测结果序列:", observations)

# # 初始化参数
# pi0 = 0.5  # 硬币 A 正面向上的初始概率
# p0 = 0.5  # 硬币 B 正面向上的初始概率
# q0 = 0.5  # 硬币 C 正面向上的初始概率
# theta = (pi0, p0, q0)  # 参数以元组形式表示
# epsilon = 1e-6  # 收敛阈值
# max_iter = 100  # 最大迭代次数
#
# # 模拟观测数据
# n = 100
# coin_A = np.random.rand(n) < pi0
# coin_B_or_C = np.where(coin_A, np.random.rand(n) < p0, np.random.rand(n) < q0)
# observations = np.where(coin_A, coin_B_or_C, 0)
#
# # EM 算法
# for _ in range(max_iter):
#     # E步：计算隐变量的后验概率
#     mu = theta[0] * p0 * (1 - p0) ** (1 - observations) / \
#          (theta[0] * p0 * (1 - p0) ** (1 - observations) +
#           (1 - theta[0]) * q0 * (1 - q0) ** (1 - observations))
#
#     # M步：更新参数
#     theta_new = (np.sum(mu) / n, np.sum(mu * observations) / np.sum(mu), (n - np.sum(mu)) / (n - np.sum(mu)))
#
#     # 检查收敛性
#     if np.linalg.norm(np.array(theta) - np.array(theta_new)) < epsilon:
#         break
#
#     # 更新参数
#     theta = theta_new
#
# # 打印估计的概率
# print("估计的硬币 A、B、C 正面向上的概率:", theta)

np.random.seed(0)
class ThreeCoinsMode(object):
    def __init__(self, n_epoch=5):
        # 运用EM算法求解三银币模型
        # :param n_epoch: 迭代次数
        self.n_epoch = n_epoch
        self.params = {'pi': None, 'p': None, 'q': None, 'mu': None}

    def __init_params(self, n):
        self.params = {'pi': [0.7],
                       'p': [0.3],
                       'q': [0.6],
                       'mu': np.random.rand(n)}

    def E_step(self, y, n):
        pi = self.params['pi'][0]
        p = self.params['p'][0]
        q = self.params['q'][0]
        for i in range(n):
            self.params['mu'][i] = (pi * pow(p, y[i]) * pow(1 - p, 1 - y[i])) / (
                        pi * pow(p, y[i]) * pow(1 - p, 1 - y[i]) + (1 - pi) * pow(q, y[i]) * pow(1 - q, 1 - y[i]))

    def M_step(self, y, n):
        mu = self.params['mu']
        self.params['pi'][0] = sum(mu) / n
        self.params['p'][0] = sum([mu[i] * y[i] for i in range(n)]) / sum(mu)
        self.params['q'][0] = sum([(1 - mu[i]) * y[i] for i in range(n)]) / sum([1 - mu_i for mu_i in mu])

    def fit(self, y):
        n = len(y)
        self.__init_params(n)
        print(0, self.params['pi'], self.params['p'], self.params['q'])
        for i in range(self.n_epoch):
            self.E_step(y, n)
            self.M_step(y, n)
            print(i + 1, self.params['pi'], self.params['p'], self.params['q'])


y = observations
tcm = ThreeCoinsMode()
tcm.fit(y)
print(tcm)

