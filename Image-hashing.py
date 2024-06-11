import cv2,os
import numpy as np
import matplotlib.pyplot as plt
#高斯低通滤波
def apply_custom_gaussian_filter(image, sigma):
    kernel_half_size = 3 // 2  # 核大小为3，因此半尺寸为1
    x, y = np.meshgrid(np.arange(-kernel_half_size, kernel_half_size + 1), 
                       np.arange(-kernel_half_size, kernel_half_size + 1))
    # 使用公式(17)计算高斯核
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    gaussian_kernel /= np.sum(gaussian_kernel)
    filtered_image = cv2.filter2D(image, -1, gaussian_kernel, borderType=cv2.BORDER_REFLECT101)

    return filtered_image


# 定义一个函数来归一化图像
def normalize(image):
    min_val = np.min(image)  # 图像最小值
    max_val = np.max(image)  # 图像最大值
    normalized_image = (image - min_val) / (max_val - min_val)  # 归一化图像
    return normalized_image

def compute_luminance_saliency_map(B, G, R):
    # 计算亮度图像
    I = (B + G + R) / 3
    normalized_cs_feature_maps = []  # 存储归一化的特征图的列表

    # 计算每个 c, s 对的特征图并归一化
    for c in range(2, 5):  # c 的值为 2, 3, 和 4
        for delta in [3, 4]:  # delta 的值为 3 和 4，所以 s 为 c+3 和 c+4
            s = c + delta
            # 计算中心和周围的特征图
            center = cv2.GaussianBlur(I, (0, 0), c)
            surround = cv2.GaussianBlur(I, (0, 0), s)
            # 计算中心-周围差异
            cs_diff = np.abs(center - surround)
            # 归一化中心-周围差异
            norm_cs_diff = normalize(cs_diff)
            normalized_cs_feature_maps.append(norm_cs_diff)

    # 将归一化的中心-周围差异特征图求和，以创建最终的显著性图
    final_saliency_map = np.sum(normalized_cs_feature_maps, axis=0)

    # 将最终的显著性图归一化为0到255之间的值
    final_saliency_map_normalized = cv2.normalize(final_saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    final_saliency_map_uint8 = np.uint8(final_saliency_map_normalized)

    return final_saliency_map_uint8

#颜色显著图
def compute_color_saliency_map(B, G, R):
    # 初始化空列表以存储每个通道和每个尺度的特征图
    rg_color_component_images = []
    by_color_component_images = []

    # 计算每个尺度的特征图
    for sigma in (0, 9):
        # 计算每个通道的颜色特征
        red = R - (G + B) / 2
        green = G - (R + B) / 2
        blue = B - (R + G) / 2
        yellow = (R + G) / 2 - np.abs(R - G) / 2 - B

        # 使用当前的 sigma 对颜色特征进行高斯模糊
        red_blurred = cv2.GaussianBlur(red, (3, 3), sigma)
        green_blurred = cv2.GaussianBlur(green, (3, 3), sigma)
        blue_blurred = cv2.GaussianBlur(blue, (3, 3), sigma)
        yellow_blurred = cv2.GaussianBlur(yellow, (3, 3), sigma)

        # 计算 rg 和 by 特征图
        rg = np.abs(cv2.subtract(red_blurred, green_blurred))
        by = np.abs(cv2.subtract(blue_blurred, yellow_blurred))
        # 调整 rg 和 by 特征图的大小
        rg_resized = cv2.resize(rg, (red_blurred.shape[1], red_blurred.shape[0]), interpolation=cv2.INTER_LINEAR)
        by_resized = cv2.resize(by, (red_blurred.shape[1], red_blurred.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 对 rg 和 by 特征图进行归一化处理
        rg_normalized = cv2.normalize(rg_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        by_normalized = cv2.normalize(by_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # 添加归一化后的颜色组成特征图到列表中
        rg_color_component_images.append(rg_normalized)
        by_color_component_images.append(by_normalized)

    # 将所有归一化的颜色组成特征图求和，创建颜色显著性图
    rg_saliency_map = np.sum(rg_color_component_images, axis=0)
    by_saliency_map = np.sum(by_color_component_images, axis=0)

    # 将颜色显著性图归一化为0到255之间的值
    rg_saliency_map_normalized = cv2.normalize(rg_saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    by_saliency_map_normalized = cv2.normalize(by_saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 将 rg_saliency_map_normalized 调整大小为与 by_saliency_map_normalized 相同的大小
    rg_saliency_map_normalized_resized = cv2.resize(rg_saliency_map_normalized, (by_saliency_map_normalized.shape[1], by_saliency_map_normalized.shape[0]))

    # 将调整大小后的颜色显著性图相加，创建最终的颜色显著性图
    color_saliency_map = rg_saliency_map_normalized_resized + by_saliency_map_normalized

    # 将颜色显著性图转换为 uint8 数据类型以显示
    color_saliency_map_uint8 = np.uint8(color_saliency_map)

    return color_saliency_map_uint8

#方向显著图
def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths

#构建Gabor滤波器
def build_filters():
    filters = []
    ksize = [7,9,11,13,15,17] # gabor尺度，6个
    lamda = np.pi/2.0         # 波长
    for theta in np.arange(0, np.pi, np.pi / 4): #gabor方向，0°，45°，90°，135°，共四个
        for K in range(6):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum() # 这里不是很理解
            filters.append(kern)
    plt.figure(1)

    #用于绘制滤波器
    for temp in range(len(filters)):
        plt.subplot(4, 6, temp + 1)
        plt.imshow(filters[temp])
    plt.show()
    return filters

# Gabor特征提取
def getGabor(img, filters):
    res = [] # 滤波结果
    for i in range(len(filters)):
        fimg = cv2.filter2D(img, cv2.CV_8UC1, filters[i])
        res.append(np.asarray(fimg, dtype=np.float32))  # 确保使用浮点数来避免溢出

    # 用于绘制滤波效果
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(4, 6, temp+1)
        plt.imshow(res[temp], cmap='gray')
    plt.show()

    return res  # 返回滤波结果,结果为24幅图，按照gabor角度排列

# 融合方向特征图
def fuse_orientation_maps(res):
    # 将所有方向特征图叠加起来，获取平均值
    sum_of_maps = np.sum(res, axis=0)
    orientation_salience_map = sum_of_maps / len(res)
    
    # 或者获取最大值
    # orientation_salience_map = np.max(res, axis=0)
    
    # 归一化显著图
    orientation_salience_map = cv2.normalize(orientation_salience_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return orientation_salience_map

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ...之前的函数定义...

if __name__ == '__main__':
    image1 = cv2.imread("lena.png")
    sigma = 1.0
    lena_img_resized = cv2.resize(image1, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = apply_custom_gaussian_filter(lena_img_resized, sigma)

    B, G, R = cv2.split(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 确保图像是灰度的
    
    color_saliency_map_uint8 = compute_color_saliency_map(B, G, R)
    luminance_saliency_map_uint8 = compute_luminance_saliency_map(B, G, R)

    filters = build_filters()
    res = getGabor(img_gray, filters)
    orientation_salience_map = fuse_orientation_maps(res)

    Ic_float = color_saliency_map_uint8.astype(np.float32)
    Cc_float = luminance_saliency_map_uint8.astype(np.float32) 
    Oc_float = orientation_salience_map.astype(np.float32)
    S = (Ic_float + Cc_float + Oc_float) / 3.0
    S_normalized = cv2.normalize(S, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 使用 matplotlib 同屏显示所有图像
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(lena_img_resized, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(color_saliency_map_uint8, cv2.COLOR_BGR2RGB)), plt.title('Color Saliency')
    plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(luminance_saliency_map_uint8, cv2.COLOR_BGR2RGB)), plt.title('Luminance Saliency')
    plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(orientation_salience_map, cv2.COLOR_BGR2RGB)), plt.title('Orientation Salience')
    plt.show()

    # 保存最终显著图到文件
    S_normalized = np.clip(S_normalized, 0, 255).astype(np.uint8)

    # 显示显著性图
    cv2.imshow("Salience Map", S_normalized)  # 显示窗口的标题和显著性图
    cv2.waitKey(0)  # 等待键盘输入
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

    # 保存显著性图到文件
    saliency_map_filename = 'salience_map.png'
    cv2.imwrite(saliency_map_filename, S_normalized)


import cv2
import numpy as np

# 读取图像
saliency_map = cv2.imread('salience_map.png', cv2.IMREAD_GRAYSCALE)  # 确保是灰度图像
if saliency_map is None:
    raise ValueError("Image not found or path is incorrect")

# 图像的尺寸和分块的尺寸
img_height, img_width = saliency_map.shape
block_size = 32  # 假设块的大小是32x32

# 检查图像尺寸是否能被块大小整除
if img_height % block_size != 0 or img_width % block_size != 0:
    raise ValueError("Image dimensions should be divisible by the block size")

# 初始化结果矩阵X
X = np.zeros((block_size**2, (img_height * img_width) // (block_size**2)), dtype=np.float32)

# 对图像进行非重叠分块，并进行Z字形扫描
block_index = 0
for y in range(0, img_height, block_size):
    for x in range(0, img_width, block_size):
        # 提取块
        block = saliency_map[y:y+block_size, x:x+block_size]
        
        # Z字形扫描块
        z_scan_block = block.flatten('F' if block_index % 2 == 0 else 'C')  # 'F'表示按列扫描，'C'按行扫描
        
        # 将扫描后的向量放入结果矩阵X
        X[:, block_index] = z_scan_block
        block_index += 1

# 转换结果矩阵X为图像并保存
X_image = (X - X.min()) / (X.max() - X.min()) * 255  # 归一化到0-255范围
X_image = X_image.astype(np.uint8)  # 转换为无符号整型

# 保存X图像
cv2.imshow('X_image.jpg', X_image)  # 显示窗口的标题和显著性图
cv2.waitKey(0)  # 等待键盘输入
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
cv2.imwrite('X_image.jpg', X_image)

print("Z-scan and reassembly complete, result saved as 'X_image.jpg'")


import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn import datasets, manifold
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D



def cal_pairwise_dist(data):
	expand_ = data[:, np.newaxis, :]
	repeat1 = np.repeat(expand_, data.shape[0], axis=1)
	repeat2 = np.swapaxes(repeat1, 0, 1)
	D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
	return D


def get_n_neighbors(data, n_neighbors=10):
	dist = cal_pairwise_dist(data)
	dist[dist < 0] = 0
	n = dist.shape[0]
	N = np.zeros((n, n_neighbors))
	for i in range(n):
		# np.argsort 列表从小到大的索引
		index_ = np.argsort(dist[i])[1:n_neighbors+1]
		N[i] = N[i] + index_
	return N.astype(np.int32)                         # [n_features, n_neighbors]


def lle(data, n_dims, n_neighbors):
	N = get_n_neighbors(data, n_neighbors)            # k近邻索引
	n, D = data.shape                                 # n_samples, n_features
	# prevent Si to small
	if n_neighbors > D:
		tol = 1e-3
	else:
		tol = 0
	# calculate W
	W = np.zeros((n_neighbors, n))
	I = np.ones((n_neighbors, 1))
	for i in range(n):                                # data[i] => [1, n_features]
		Xi = np.tile(data[i], (n_neighbors, 1)).T     # [n_features, n_neighbors]
		                                              # N[i] => [1, n_neighbors]
		Ni = data[N[i]].T                             # [n_features, n_neighbors]
		Si = np.dot((Xi-Ni).T, (Xi-Ni))               # [n_neighbors, n_neighbors]
		Si = Si + np.eye(n_neighbors)*tol*np.trace(Si)
		Si_inv = np.linalg.pinv(Si)
		wi = (np.dot(Si_inv, I)) / (np.dot(np.dot(I.T, Si_inv), I)[0,0])
		W[:, i] = wi[:,0]
	W_y = np.zeros((n, n))
	for i in range(n):
		index = N[i]
		for j in range(n_neighbors):
			W_y[index[j],i] = W[j,i]
	I_y = np.eye(n)
	M = np.dot((I_y - W_y), (I_y - W_y).T)
	eig_val, eig_vector = np.linalg.eig(M)
	index_ = np.argsort(np.abs(eig_val))[1:n_dims+1]
	Y = eig_vector[:, index_]
	return Y



# def process_image_with_lle(image, n_neighbors=15, n_dims=40, block_size=32):
#     # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError("Image not found or path is incorrect")

#     height, width = image.shape
#     # 确保图像可以被块大小整除
#     assert height % block_size == 0 and width % block_size == 0, "Image dimensions must be divisible by the block size"

#     # 初始化LLE
#     lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_dims, method='standard', eigen_solver='auto')

#     # 分块并将块展平
#     blocks = [image[i:i+block_size, j:j+block_size].flatten() 
#               for i in range(0, height, block_size) 
#               for j in range(0, width, block_size)]

#     # 转换块为NumPy数组
#     blocks = np.array(blocks)

#     # 应用LLE算法
#     Y = lle.fit_transform(blocks)

#     # 返回降维后的数据
#     return Y


if __name__ == '__main__':
    # 设置图像路径
    image_path = 'lena_DCT-generated.png'  # 调整为正确的图像路径
    # 设置LLE参数
    n_neighbors = 15
    n_dims = 40
    block_size = 32  # 块的大小

    # 处理图像并获取LLE降维结果
    # Y = lle(X_image, n_neighbors=n_neighbors, n_dims=n_dims, block_size=block_size)
    Y= lle(X_image, n_dims=40, n_neighbors=15)
    print(Y)

    # 计算 Y 的每一维的均值
    mu = np.mean(Y, axis=0)
    
    # 计算每个维度的方差
    delta_squared = np.sum((Y - mu) ** 2, axis=0) / (Y.shape[0] - 1)
    
    # 计算每个维度的特征统计量 H(i)
    H = np.round(delta_squared * 1000).astype(int)
    
    # 打印方差和特征统计量
    print("方差: ", delta_squared)
    print("LLL向量的统计特征 H(i): ", H)
    
    # 显示降维结果
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(Y[:, 0], Y[:, 1], c='blue', marker='.')
    plt.title('LLE Reduction of Image Data')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
