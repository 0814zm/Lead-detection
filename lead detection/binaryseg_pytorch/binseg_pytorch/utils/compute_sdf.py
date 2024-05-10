import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    # img_gt = img_gt.astype(np.uint8)  # 转换为uint8
    # print(f"img_gt:{img_gt}")
    # normalized_sdf = np.zeros(out_shape)  # sdf初始化为输入同样大小
    #
    # for b in range(out_shape[0]):  # batch size
    #     posmask = img_gt[b].astype(bool)
    #     print(f"posmask:{posmask}")
    #     if posmask.any():  # 对矩阵所有元素做或运算，存在True则返回True(判断是否存在前景像素，若存在，执行下列操作)
    #         negmask = ~posmask  # 将第一个batch的所有像素点取反，~表示布尔值取反
    #         print(f"negmask:{negmask}")
    #         posdis = distance(posmask)  # scipy.ndimage.distance_transform_edt, 结果为每个前景像素点1到最近的背景0的距离，float64
    #         print(f"posdis:{posdis}")
    #         print(posdis.dtype)
    #         negdis = distance(negmask)  # 将原图像取反再求的距离
    #         print(f"negdis:{negdis}")
    #         boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)  # skimage.segmentation.find_boundaries
    #         print(f"boundary:{boundary}")  # 背景像素不变，仍为0，前景像素1若上下左右均为前景1，则变为0，否则仍为1
    #         print(f"negdis归一化后:{(negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis))}")
    #         print(f"posdis归一化后:{(posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))}")
    #         sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))  # 分别对posdis和negdis归一化到[0,1],然后相减,也实现了将sdf归一化到【-1，1】
    #         print(f"sdf:{sdf}")
    #         sdf[boundary==1] = 0 # 把sdf中对应边界处均变为0值
    #         print(f"sdf1:{sdf}")
    #         normalized_sdf[b] = sdf
    #         # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
    #         # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    #     print(f"normalized_sdf:{normalized_sdf}")

    img_gt = img_gt.astype(np.uint8)  # 转换为uint8
    # print(f"img_gt:{img_gt}")
    normalized_sdf = np.zeros(out_shape)  # sdf初始化为输入同样大小

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(bool)
        # print(f"posmask:{posmask}")
        if posmask.any():  # 对矩阵所有元素做或运算，存在True则返回True(判断是否存在前景像素，若存在，执行下列操作)
            negmask = ~posmask  # 将第一个batch的所有像素点取反，~表示布尔值取反
            # print(f"negmask:{negmask}")
            posdis = distance(posmask)  # scipy.ndimage.distance_transform_edt, 结果为每个前景像素点1到最近的背景0的距离，float64
            # print(f"posdis:{posdis}")
            # print(posdis.dtype)
            negdis = distance(negmask)  # 将原图像取反再求的距离
            # print(f"negdis:{negdis}")
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)  # skimage.segmentation.find_boundaries
            # print(f"boundary:{boundary}")  # 背景像素不变，仍为0，前景像素1若上下左右均为前景1，则变为0，否则仍为1
            # print(f"negdis归一化后:{(negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis))}")
            # print(f"posdis归一化后:{(posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))}")
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis))  # 分别对posdis和negdis归一化到[0,1],然后相减,也实现了将sdf归一化到【-1，1】
            # print(f"sdf:{sdf}")
            sdf[boundary == 1] = 0  # 把sdf中对应边界处均变为0值
            # print(f"sdf1:{sdf}")
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
        # print(f"normalized_sdf:{normalized_sdf}")

    return normalized_sdf
if __name__ == '__main__':
    a = np.zeros((4,1,5,5))
    b = [[[[0, 0, 0, 1, 1],
          [0, 1, 0, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 1, 0],
          [1, 1, 1, 1, 0]],

          [[1, 0, 0, 1, 1],
           [0, 0, 1, 1, 0],
           [1, 0, 0, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 0]],

          [[0, 1, 0, 1, 1],
           [0, 1, 0, 0, 1],
           [1, 0, 1, 1, 0],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 0]]],


        [[[0, 0, 0, 1, 1],
          [1, 1, 1, 1, 1],
          [1, 0, 0, 1, 0],
          [1, 1, 0, 1, 1],
          [1, 1, 1, 1, 0]],

          [[1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [1, 1, 0, 0, 1],
           [1, 0, 0, 1, 1],
           [1, 1, 1, 1, 0]],

          [[1, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 0]]]]
    b = np.array(b)  # int32
    print(b.dtype)
    print(a[:, 0, ...].shape)
    # sdf = compute_sdf(b, (2,3,5,5))