import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_path, keep_ratio):
    # 读取图片并转换为灰度图
    image = Image.open(image_path).convert("L")
    image_matrix = np.array(image)

    # 进行奇异值分解
    U, S, Vt = np.linalg.svd(image_matrix)

    # 计算保留的奇异值个数
    keep_num = int(len(S) * keep_ratio)

    # 构造奇异值矩阵
    S_matrix = np.zeros((U.shape[1], Vt.shape[0]))
    S_matrix[:keep_num, :keep_num] = np.diag(S[:keep_num])

    # 重构图像
    reconstructed_matrix = U.dot(S_matrix).dot(Vt)
    reconstructed_image = Image.fromarray(reconstructed_matrix.astype(np.uint8))

    return reconstructed_image

# 设置保留奇异值的比例列表
keep_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

# 原始图片路径
image_path = "C:/Users/lmx/Desktop/图片1.jpg"
plt.rcParams['font.family'] = 'SimHei'
# 展示原始图片
original_image = Image.open(image_path)
plt.subplot(3, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('原始图片')

# 逐个保留奇异值并重构图片
for i, keep_ratio in enumerate(keep_ratios):
    reconstructed_image = compress_image(image_path, keep_ratio)
    plt.subplot(3, 3, i+2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'保留比例: {keep_ratio * 100}%')

plt.tight_layout()
plt.show()