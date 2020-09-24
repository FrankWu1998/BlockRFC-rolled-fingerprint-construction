__author__ = 'apple'

import numpy as np
import cv2
import copy
import datetime
import os
from imageio import imwrite
from pathlib import Path

# ----------------------------------------------------------------------------
# 模块一：重命名指纹图像
# ----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 90
count = 0
# fileNumber 文件的序号
for fileNumber in range(51, dirNum + 1):
    projectPath = "/Users/yifanwu/Papers_Frank/2020-IEEE-Access--Rolled-Fingerprint-Construction/"
    path = projectPath + "Experiments/datasets/finger_data/" + str(fileNumber)  # 测试集的文件根目录
    names = os.listdir(path)
    for old_name in names:
        # print(old_name[0:3])
        new_name = 'Bitbmp00' + old_name[0:3] + '.bmp'
        # new_name = "Bitbmp0" + old_name
        # new_name = "Bitbmp00" + old_name[7:14]
        # new_name = "Bitbmp00" + old_name[14:21]
        os.rename(os.path.join(path, old_name), os.path.join(path, new_name))
