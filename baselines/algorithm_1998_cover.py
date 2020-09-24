__author__ = 'apple'
import numpy as np
import cv2
import copy
import datetime
import os
from imageio import imwrite
from pathlib import Path

# ----------------------------------------------------------------------------
# 模块一：覆盖法
#----------------------------------------------------------------------------
# 400 * 501
# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 90
count = 0
# fileNumber 文件的序号
for fileNumber in range(11, dirNum+1):
    projectPath = "/Users/yifanwu/Papers_Frank/2020-IEEE-Access--Rolled-Fingerprint-Construction/"
    path = projectPath + "Experiments/datasets/finger_data/" + str(fileNumber)  # 测试集的文件根目录
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
    img_dir = path + "/" + ("Bitbmp00%03d.bmp" % k_down)
    img0 = cv2.imread(img_dir, 0)
    img0_float = img0.astype(np.float32)
    for k in range(k_down + 1, k_upper + 1):
        img_dir = path + "/" + ("Bitbmp00%03d.bmp" % k)
        img = cv2.imread(img_dir, 0)

        # 指纹是否出现
        img_float = img.astype(np.float32)
        res_img = abs(img0_float - img_float)

        [row,col] = [img0.shape[0],img0.shape[1]]

        for i in range(0,row):
            for j in range(0,col):
                if img0[i][j] > img[i][j]: # 覆盖法，选择灰度值低的
                    img0[i][j] = img[i][j]

    # 输出滚动指纹目前为止，拼接块的选择结果

    imwrite(projectPath + 'Experiments/results/algorithm_4/' + str(fileNumber) + '_0' + '.bmp', img0)
    print(str(fileNumber) + "finish！")