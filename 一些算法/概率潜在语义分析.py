import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# 从文档中读取新闻数据
with open('xjtunews.txt', 'r', encoding='utf-8') as file:
    documents = file.readlines()

# 读取停用词表
with open('停用词.txt', 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file.readlines()])

# 定义一个函数去除空格和停用词
def preprocess_text(text, stopwords):
    # 去除空格和换行符
    text = re.sub(r'\s+', ' ', text).strip()
    # 分词
    words = jieba.lcut(text)
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    return words


# 对每篇文档进行预处理
texts = [" ".join(preprocess_text(document, stopwords)) for document in documents]
vectorizer = CountVectorizer(max_df=0.5, min_df=5, max_features=1000)
X=vectorizer.fit_transform(texts)
vocabulary=vectorizer.get_feature_names_out()

class PLSA:
    def __init__(self, K, max_iter):
        self.K = K
        self.max_iter = max_iter
    def fit(self, X):
        n_d, n_w = X.shape
        # P(z/w, d)
        p_z_dw = np.zeros((n_d, n_w, self.K))
        # P(z/d)
        p_z_d = np.random.rand(n_d, self.K)
        # P(w/z)
        p_w_z = np.random.rand(self.K, n_w)
        for i_iter in range(self.max_iter):
            # E step
            for di in range(n_d):
                for wi in range(n_w):
                    sum_zk = np.zeros((self.K))
                    for zi in range(self.K):
                        sum_zk[zi] = p_z_d[di, zi] * p_w_z[zi, wi]
                    sum1 = np.sum(sum_zk)
                    if sum1 == 0:
                        sum1 = 1
                    for zi in range(self.K):
                        p_z_dw[di, wi, zi] = sum_zk[zi] / sum1
            # M step
            # update P(z/d)

            for di in range(n_d):
                for zi in range(self.K):
                    sum1 = 0.
                    sum2 = 0.
                    for wi in range(n_w):
                        sum1 = sum1 + X[di, wi] * p_z_dw[di, wi, zi]
                        sum2 = sum2 + X[di, wi]
                    if sum2 == 0:
                        sum2 = 1
                    p_z_d[di, zi] = sum1 / sum2
            # update P(w/z)
            for zi in range(self.K):
                sum2 = np.zeros((n_w))
                for wi in range(n_w):
                    for di in range(n_d):
                        sum2[wi] = sum2[wi] + X[di, wi] * p_z_dw[di, wi, zi]
                sum1 = np.sum(sum2)
                if sum1 == 0:
                    sum1 = 1
                for wi in range(n_w):
                    p_w_z[zi, wi] = sum2[wi] / sum1
        return p_w_z, p_z_d

    def accelerated_fit(self, X):
        n_d, n_w = X.shape
        X = (X.A).astype(float)
        # P(z/w, d)
        p_z_dw = np.zeros((n_d, n_w, self.K)).astype(float)
        # P(z/d)
        p_z_d = np.random.rand(n_d, self.K).astype(float)
        # P(w/z)
        p_w_z = np.random.rand(self.K, n_w).astype(float)

        for i_iter in range(self.max_iter):
            # E step:
            numerator = np.einsum('ij, ki->kj', p_w_z, p_z_d)
            denominator = np.einsum('kji->kj', numerator)
            denominator = np.expand_dims(denominator, 2).repeat(self.K, axis=2)
            denominator[denominator == 0] = 1
            p_z_dw = numerator / denominator
        # M step:
            numerator = np.einsum('ji, jik->jk', X, p_z_dw)
            denominator = np.einsum('ji->j', X)
            denominator = np.expand_dims(denominator, 1).repeat(self.K, axis=1)
            denominator[denominator == 0] = 1
            p_z_d = numerator / denominator
        return p_w_z,p_z_d

n_topic=5
plsa=PLSA(n_topic,50)
p_w_z,p_z_d=plsa.fit(X)
for i in range(n_topic):
    print("topic %d"%(i+1))
    top=p_w_z[i].argsort()[::-1][0:10]
    for j in top:
        print(vocabulary[j])
    print()





