__author__ = 'apple'
import numpy as np
import cv2
import copy
import os
from imageio import imwrite
from pathlib import Path

# ----------------------------------------------------------------------------
# 模块一：基于关键列的实时指纹拼接
#----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 90
# fileNumber 文件的序号
for fileNumber in range(11, dirNum + 1):
    K = 300
    layer_Row = 1  # 越大越矮
    layer_Column = 800  # 越大越细
    projectPath = "/Users/yifanwu/Papers_Frank/2020-IEEE-Access--Rolled-Fingerprint-Construction/"
    path = projectPath + "Experiments/datasets/finger_data/" + str(fileNumber)  # 测试集的文件根目录
    maxImgNum = 300  # 最大文件数量
    st = maxImgNum  # 文件开始的编号
    ed = 0  # 文件终止的编号
    names = os.listdir(path)
    k_upper = -10000
    k_down = 10000
    for old_name in names:
        num = int(old_name[7:11])
        if k_upper < num:
            k_upper = num
        if k_down > num:
            k_down = num
    print(k_upper)
    print(k_down)
    path = path + "/"

    st = k_down # 文件开始的编号
    ed = k_upper  # 文件终止的编号

    # 背景图像
    img_name = path + ("Bitbmp000%02d.bmp" % st)
    # print(img_name)
    img0 = cv2.imread(img_name, 0)
    img0_float = img0.astype(np.float32)
    sizeRow, sizeColumn = img0.shape[0], img0.shape[1]  # 行数 列数 640 640
    # 将图像进行分块
    w_x = sizeRow // layer_Row  # 640
    w_y = sizeColumn // layer_Column  # 1

    # 创建K个 640 * 640 的整型零矩阵
    val_a = np.zeros((K, sizeRow, sizeColumn), dtype=np.int)

    # 创建K个 16+1 * 128+1 的浮点型零矩阵
    f = np.zeros((K, layer_Row + 1, layer_Column + 1), dtype=np.float32)
    c = np.zeros((K, layer_Row + 1, layer_Column + 1), dtype=np.float32)

    # 创建1个 K * 2 的整型零矩阵
    cer = np.zeros((K, 2), dtype=np.int)

    # 创建1个 16 * 128 的整型零矩阵
    result_vis = np.zeros((layer_Row, sizeColumn), dtype=np.int)

    # 用于记录那些帧图像被拼接过程使用
    useful = np.zeros(K)


    # 基于灰度二分法，寻找中心列
    def find_column(image, begin, end):
        tag = sizeRow * sizeColumn * 255
        keyCol = -1
        for r in range(begin+1,end-1):
            val_three = np.sum( image[:, r-1:r+1] ) # 3列灰度和
            if val_three < tag:
                tag = val_three
                keyCol = r
        return keyCol


    left_to_right = 1  # 默认指纹从左滚动到右
    start = -1  # 记录开始指纹拼接的第一张图像的序号
    op_flag = 0
    bound1 = 64 + 5
    bound2 = 0 - 5
    for k in range(st + 1, ed + 1):
        img_dir = path + ("Bitbmp00%03d.bmp" % k)
        img = cv2.imread(img_dir, 0)

        # 指纹是否出现
        img_float = img.astype(np.float32)
        res_img = abs(img0_float - img_float)
        cnt1 = np.sum(res_img > 25)  # 指纹点数
        if cnt1 < 15555:
            continue

        if start == -1:
            start = k  # 记录出现的第一张图片序号

        useful[k] = 1  # 标记该图像被使用
        # 图像归一化
        img_a = np.zeros(img.shape, dtype=np.float32)
        cv2.normalize(img, img_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_a = np.uint8(img_a * 255.0)
        val_a[k] = copy.deepcopy(img_a)

        # 计算质心
        cer[k][0] = 1
        cer[k][1] = find_column(img_a, 0, layer_Column - 1)

        # 计算图像块的背景点数和灰度值之和
        for j in range(0, layer_Column):
            for i in range(0, layer_Row):
                tmp_img = img_a[i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y].astype(np.float32)
                f[k][i][j] = np.sum(tmp_img)
                c[k][i][j] = np.sum(tmp_img > 240)

        if bound2 >= cer[k][1] >= bound1:
            op_flag = op_flag + 1
        else:
            op_flag = 0
            if bound1 > cer[k][1]:
                bound1 = cer[k][1]
            if bound2 < cer[k][1]:
                bound2 = cer[k][1]

        # 从左到右滚动
        if left_to_right == 1:
            for i in range(0, layer_Row):  # 从0～layer_Row-1
                # 若该行的质心块未被覆盖，或者已覆盖的该区域块灰度和比当前帧的大，则选择当前帧。
                if result_vis[i][cer[k][1]] == 0 or f[result_vis[i][cer[k][1]]][i][cer[k][1]] > f[k][i][cer[k][1]]:
                    result_vis[i][cer[k][1]] = k

                for j in range(cer[k][1], 0, -1):  # 从cer[k][1] ～ 1
                    if result_vis[i][j - 1] == 0:
                        result_vis[i][j - 1] = k
                    elif f[result_vis[i][j - 1]][i][j - 1] > f[k][i][j - 1]:
                        result_vis[i][j - 1] = k
                    else:
                        break
                for j in range(cer[k][1], layer_Column-1):
                    if result_vis[i][j + 1] == 0:
                        result_vis[i][j + 1] = k
                    elif f[result_vis[i][j + 1]][i][j + 1] > f[k][i][j + 1]:
                        result_vis[i][j + 1] = k

        for j in range(0, layer_Column):
            for i in range(0, layer_Row):
                img[i * w_x:(i + 1) * w_x,
                j * w_y:(j + 1) * w_y] = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x,
                                         j * w_y:(j + 1) * w_y]

        img0 = img

    # 输出滚动指纹目前为止，拼接块的选择结果

    imwrite(projectPath + 'Experiments/results/algorithm_3/' + str(fileNumber) + '_0' + '.bmp', img0)
    print(str(fileNumber) + "finish！")