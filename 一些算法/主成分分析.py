import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

# 读取图片并转换为灰度图
img = Image.open('C:/Users/lmx/Desktop/图片1.jpg').convert('L')
img_array = np.array(img)

# 将像素值归一化到0-1之间
img_array = img_array / 255.0

# 将图片展平为1D向量
img_flat = img_array.flatten()

# 对图片进行PCA压缩
pca = PCA()
pca.fit(img_flat.reshape(-1, 1))

# 按照给定的保留比例计算需要保留的主成分个数
retain_ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
n_components_list = [int(len(pca.components_) * ratio) for ratio in retain_ratios]

# 重构图片并与原图片进行比较
reconstructed_images = []

# 展示原始图片
image_path = "C:/Users/lmx/Desktop/图片1.jpg"
plt.rcParams['font.family'] = 'SimHei'
original_image = Image.open(image_path)
plt.subplot(3, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图片')

for i, n_components in enumerate(n_components_list):
    pca.n_components = n_components

    # 对图片进行压缩和重构
    compressed_img = pca.transform(img_flat.reshape(-1, 1))
    reconstructed_img = pca.inverse_transform(compressed_img).reshape(img_array.shape)
    # 将重构的图片存储起来
    reconstructed_images.append(reconstructed_img)

    plt.subplot(3, 3, i + 2)
    # 绘制重构后的图片
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(f'保留比例({retain_ratios[i]*100}%)')

# 显示所有图片
plt.tight_layout()
plt.show()

