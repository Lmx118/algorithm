import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 导入人脸数据集
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# 获取数据集中的图像数据和对应的标签
X = lfw_people.data
y = lfw_people.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 创建支持向量机分类器对象
svm = SVC()

# 在原始数据上训练支持向量机分类器
svm.fit(X_train, y_train)
score_original = svm.score(X_test, y_test)

# 使用PCA将数据降维，并选择最重要的12个主成分
pca = PCA(n_components=12)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 在降维后的数据上训练支持向量机分类器
svm.fit(X_train_pca, y_train)
score_pca = svm.score(X_test_pca, y_test)

print("使用原始数据的分类准确率：", score_original)
print("经过PCA降维后的分类准确率：", score_pca)

# 绘制前12个主成分对应的特征脸
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(lfw_people.images.shape[1], lfw_people.images.shape[2]), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC {i+1}")
plt.show()