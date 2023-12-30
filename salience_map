import cv2,os
import numpy as np
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    filters = build_filters()
    image=cv2.imread("lena.png")
    B, G, R = cv2.split(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 确保图像是灰度的
    
    color_saliency_map_uint8 = compute_color_saliency_map(B, G, R)
    cv2.imshow('C', color_saliency_map_uint8)

    luminance_saliency_map_uint8 = compute_luminance_saliency_map(B, G, R)
    cv2.imshow('I', luminance_saliency_map_uint8)

    res = getGabor(img_gray, filters)
    orientation_salience_map = fuse_orientation_maps(res)
    cv2.imshow("O",orientation_salience_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Ic_float = color_saliency_map_uint8.astype(np.float32)
    Cc_float = luminance_saliency_map_uint8.astype(np.float32) 
    Oc_float = orientation_salience_map.astype(np.float32)
    S = (Ic_float + Cc_float + Oc_float) / 3.0
    S_normalized = cv2.normalize(S, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Salience Map', S_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存最终显著图到文件
    saliency_map_filename = 'salience_map.png'
    cv2.imwrite(saliency_map_filename, S_normalized)
