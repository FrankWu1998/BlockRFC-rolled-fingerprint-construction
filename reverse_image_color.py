__author__ = 'apple'
import numpy as np
import cv2
import copy
import datetime
import os
from imageio import imwrite
from pathlib import Path
from PIL import Image

# ----------------------------------------------------------------------------
# 模块一：给算法结果图形重命名
# 输入：结果图形集合
# 输出：有效名称的算法结果集合
# 备注：第1、10、16都是同一个手指
#----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 4
count = 0
# fileNumber 文件的序号
for fileNumber in range(1, dirNum+1):
    K = 300
    layer_Row = 8  # 越大越矮
    layer_Column = 64  # 越大越细
    projectPath = "/Users/apple/PycharmProjects/paper2020_fingerprint/"
    path = projectPath + "FAPImages/" + str(fileNumber)  # 测试集的文件根目录
    names = os.listdir(path)
    for old_name in names:
        loc = path + "/" + old_name
        img0 = cv2.imread(loc)
        row,col = img0.shape[0],img0.shape[1]
        for r in range(0,row):
            for p in range(0,col):
                img0[r][p] = 255 - img0[r][p]
        imwrite(loc,img0)