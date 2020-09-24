__author__ = 'apple'
import numpy as np
import cv2
import copy
import datetime
import os
from imageio import imwrite
from pathlib import Path

# ----------------------------------------------------------------------------
# 模块一：过滤非指纹图像
# 输入：采集得到的指纹图形序列
# 输出：有效指纹序列
# 备注：983张有效源图像、18个文件夹
# ----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
projectPath = "/Users/apple/PycharmProjects/paper2020_fingerprint/"
path = projectPath + "FAPImages/finger20200226/D/img9/"  # 测试集的文件根目录
dirNum = 50
count = 0
# fileNumber 文件的序号
fileNumber = 40 #过滤到目标文件夹
kkk = 0

for fi in range(0,fileNumber+1):
    tmp_file = Path(path + str(fi))
    if tmp_file.exists(): #文件夹不为空
        K = 300
        maxImgNum = 300  # 最大文件数量
        st = maxImgNum  # 文件开始的编号
        ed = 0  # 文件终止的编号
        path1 = path + str(fi) + "/"

        for i in range(0, maxImgNum):
            img_name_tmp = path1 + ("%03d.png" % i)
            # print(img_name_tmp)
            my_file = Path(img_name_tmp)
            if my_file.exists():
                if i < st:
                    st = i
                if i > ed:
                    ed = i

        if st == maxImgNum: #文件为空
            continue

        # 背景图像
        img_name = path1 + ("%03d.png" % st)
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
        tmp_dir_path = projectPath + "FAPImages/effective_data/" + str(fileNumber)
        # os.mkdir(tmp_dir_path)
        for k in range(st + 1, ed + 1):
            img_dir = path1 + ("%03d.png" % k)
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
        print("Next is " + str(kkk))

