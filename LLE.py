
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from sklearn import datasets, manifold
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
from numpy.linalg import norm



def cal_pairwise_dist(data):
	expand_ = data[:, np.newaxis, :]
	repeat1 = np.repeat(expand_, data.shape[0], axis=1)
	repeat2 = np.swapaxes(repeat1, 0, 1)
	D = np.linalg.norm(repeat1 - repeat2, ord=2, axis=-1, keepdims=True).squeeze(-1)
	return D

#选择邻接点

def get_n_neighbors(data, n_neighbors=10):
    dist = cal_pairwise_dist(data)
    # 确保没有负值的距离
    dist[dist < 0] = 0
    n = dist.shape[0]
    N = np.zeros((n, n_neighbors), dtype=int)
    for i in range(n):
        # np.argsort 返回数组排序后的索引
        index_ = np.argsort(dist[i])[1:n_neighbors+1]  # 排除距离最小（即自己）的索引
        N[i, :] = index_  # 直接赋值，这里确保N[i, :]已经初始化为正确的形状
    return N

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

# 根据图片中的公式计算重建权重W
def calculate_weights(X, neighbors_indices, n_neighbors):
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))

    # 定义优化问题的目标函数
    def reconstruction_error(w, Z, xi):
        return np.linalg.norm(xi - np.dot(w, Z), 2)**2
    
    # 等式约束，权重之和必须为 1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # 对每个点计算权重
    for i in range(n_samples):
        xi = X[i]
        nbrs_indices = neighbors_indices[i]  # 使用已经计算好的邻居索引
        Z = X[nbrs_indices] - xi  # 邻居点相对于xi的位置
        
        # 初始权重，等分权重
        w_init = np.full(n_neighbors, 1.0 / n_neighbors)
        
        # 优化权重
        res = minimize(reconstruction_error, w_init, args=(Z, xi),
                       constraints=cons, method='SLSQP')
        
        # 保存权重
        W[i, nbrs_indices] = res.x

    return W

def optimize_embedding(W, n_dims):
    n_samples = W.shape[0]
    print(n_samples)
    Y_init = np.random.rand(n_samples * n_dims)
    print(Y_init)
    

    def Phi(Y_flat):
        Y = Y_flat.reshape((n_samples, n_dims))
        Phi_Y=np.sum(norm(Y - np.dot(W, Y), axis=1) ** 2)
        return Phi_Y


    def constraint_mean(Y):
        Y = Y.reshape((n_samples, n_dims))
        return np.mean(Y, axis=0)
    
    def constraint_covariance(Y):
        Y = Y.reshape((n_samples, n_dims))
        covariance_matrix = np.cov(Y.T)  # 注意使用 .T 来得到正确的协方差矩阵
        I = np.identity(n_dims)
        return norm(covariance_matrix - I)

    constraints = [
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_covariance}
    ]

    result = minimize(Phi, Y_init, constraints=constraints, method='SLSQP', options={'disp': True})

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    Y_optimal = result.x.reshape((n_samples, n_dims))
    return Y_optimal


if __name__ == '__main__':
    # 加载图像并转换为灰度图像
    image = cv2.imread("X_image.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # 确保图像尺寸是256x256
    if image.shape != (256, 256):
        image = cv2.resize(image, (256, 256))
    # 将图像分割成32x32的块，并拉平每个块
    image_blocks = [image[i:i+32, j:j+32].flatten() 
                    for i in range(0, 256, 32) 
                    for j in range(0, 256, 32)]

    # 转换块为NumPy数组
    image_blocks = np.array(image_blocks)
    # 设置LLE算法的参数
    n_neighbors = 15  # 邻居的数量
    n_components = 40  # 降维后的维数


    neighbors_indices = get_n_neighbors(image_blocks, n_neighbors)
    print(neighbors_indices)

    # 步骤2: 计算重建权重W
    W = calculate_weights(image_blocks, neighbors_indices, n_neighbors)
    print(W)

    # 使用重建权重矩阵W来优化低维嵌入Y
    Y = optimize_embedding(W, n_components)
    print(Y)

    # 打印优化后的低维嵌入Y，或者进一步进行分析或可视化
    print("Optimized low-dimensional embedding Y:")
    print(Y)

    # 可以进一步将Y用于可视化或作为其他机器学习任务的输入
    # 例如，使用matplotlib进行可视化
    if n_components == 2:
        plt.scatter(Y[:, 0], Y[:, 1], c='blue', marker='.')
        plt.title('2D visualization of the LLE embedding')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='blue', marker='.')
        ax.set_title('3D visualization of the LLE embedding')
        plt.show()
