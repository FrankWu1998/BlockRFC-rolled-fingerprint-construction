__author__ = 'apple'
import numpy as np
import cv2
import copy
import datetime
import os
from imageio import imwrite
from pathlib import Path

# ----------------------------------------------------------------------------
# 模块一：过滤非指纹图像和低质量指纹图像
#----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 50
count = 0
# fileNumber 文件的序号
for fileNumber in range(11, dirNum + 1):
    K = 300
    projectPath = "/Users/apple/PycharmProjects/paper2020_fingerprint/"
    path = projectPath + "FAPImages/finger_DataBase/"  # 测试集的文件根目录
    maxImgNum = 300  # 最大文件数量
    st = maxImgNum  # 文件开始的编号
    ed = 0  # 文件终止的编号
    path = path + str(fileNumber) + "/"

    for i in range(0, maxImgNum):
        img_name_tmp = path + ("Bitbmp00%03d.bmp" % i)
        my_file = Path(img_name_tmp)
        if my_file.exists():
            if i < st:
                st = i
            if i > ed:
                ed = i

    # 背景图像
    img_name = path + ("Bitbmp00%03d.bmp" % st)
    print(img_name)
    img0 = cv2.imread(img_name, 0)
    img0_float = img0.astype(np.float32)
    sizeRow, sizeColumn = img0.shape[0], img0.shape[1]  # 行数 列数 640 640

    # 创建K个 640 * 640 的整型零矩阵
    val_a = np.zeros((K, sizeRow, sizeColumn), dtype=np.int)

    # 用于记录那些帧图像被拼接过程使用
    useful = np.zeros(K)

    left_to_right = 1  # 默认指纹从左滚动到右
    start = -1  # 记录开始指纹拼接的第一张图像的序号
    op_flag = 0
    bound1 = 64 + 5
    bound2 = 0 - 5
    tmp_dir_path = projectPath + "FAPImages/effective_data/" + str(fileNumber)
    os.mkdir(tmp_dir_path)
    kkk = 0
    for k in range(st + 1, ed + 1):
        img_dir = path + ("Bitbmp00%03d.bmp" % k)
        img = cv2.imread(img_dir, 0)

        # 指纹是否出现
        img_float = img.astype(np.float32)
        res_img = abs(img0_float - img_float)
        cnt1 = np.sum(res_img > 25)  # 背景点数
        if cnt1 < 15555:
            continue
        name = str(fileNumber) + "_" + str(kkk) + ".bmp"
        tmp_img_dir = tmp_dir_path + "/" + name
        imwrite(tmp_img_dir, img)
        print("ok")
        count = count + 1
        print(count)
        kkk = kkk + 1

