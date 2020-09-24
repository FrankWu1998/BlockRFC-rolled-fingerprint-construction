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
# 模块一：调整指纹图像的尺寸
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
        img0 = Image.open(loc)
        img = img0.resize((400,500))
        img.save(loc,'BMP')