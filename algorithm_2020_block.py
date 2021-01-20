__author__ = 'apple'
import numpy as np
import cv2
import copy
from imageio import imwrite
import os

# ----------------------------------------------------------------------------
# 模块一：滚动指纹拼接
# 输入：多张指纹图像
# 输出：拼接完成的一张指纹图像
#----------------------------------------------------------------------------

# 640 320 160 128 80 40 20 10 5 4 2 1
dirNum = 90
# fileNumber 文件的序号
for fileNumber in range(11, dirNum+1):
    projectPath = ""
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

    K = 300
    layer_Row = 10  # 越大越矮
    layer_Column = 80  # 越大越细

    maxImgNum = 300  # 最大文件数量

    path = path + "/"

    st = k_down # 文件开始的编号
    ed = k_upper  # 文件终止的编号

    # 背景图像
    img_name = path + ("Bitbmp000%02d.bmp" % st)
    # print(img_name)
    img0 = cv2.imread(img_name, 0)
    img0_float = img0.astype(np.float32)
    sizeColumn,sizeRow = img0.shape[0], img0.shape[1]  # 行数 列数 400 500
    # 将图像进行分块
    w_x = sizeRow // layer_Row  # 80
    w_y = sizeColumn // layer_Column  # 10
    print(w_x)
    print(w_y)
    # 创建K个 400 * 500 的整型零矩阵
    val_a = np.zeros((K, sizeColumn , sizeRow), dtype=np.int)

    # 创建K个 16+1 * 128+1 的浮点型零矩阵
    f = np.zeros((K, layer_Row + 1, layer_Column + 1), dtype=np.float32)
    c = np.zeros((K, layer_Row + 1, layer_Column + 1), dtype=np.float32)

    # 创建1个 K * 2 的整型零矩阵
    cer = np.zeros((K, 2), dtype=np.int)

    # 创建1个 16 * 128 的整型零矩阵
    result_vis = np.zeros((layer_Row, sizeColumn), dtype=np.int)

    # 用于记录那些帧图像被拼接过程使用
    useful = np.zeros(K)

    # 基于灰度二分法，寻找中心行
    def find_row(image, begin, end):
        if begin == end:
            return begin
        mid = (begin + end) // 2
        val_up = np.sum(img[begin * w_x: (mid + 1) * w_x - 1, :].astype(np.float32))
        val_down = np.sum(img[(mid + 1) * w_x: (end + 1) * w_x - 1, :].astype(np.float32))
        if val_up < val_down:
            return find_row(image, begin, mid)
        else:
            return find_row(image, mid + 1, end)


    # 基于灰度二分法，寻找中心列
    def find_column(image, begin, end):
        if begin == end:
            # print(begin)
            return begin
        mid = (begin + end) // 2
        val_left = np.sum(img[:, begin * w_y: (mid + 1) * w_y - 1])
        val_right = np.sum(img[:, (mid + 1) * w_y: (end + 1) * w_y - 1])
        if val_left < val_right:
            return find_column(image, begin, mid)
        else:
            return find_column(image, mid + 1, end)


    left_to_right = 1  # 默认指纹从左滚动到右
    start = -1  # 记录开始指纹拼接的第一张图像的序号
    op_flag = 0
    bound1 = 64 + 5
    bound2 = 0 - 5
    for k in range(st + 1, ed + 1):
        img_dir = path + ("Bitbmp00%03d.bmp" % k)
        img = cv2.imread(img_dir, 0)
        equ = cv2.equalizeHist(img) # 均衡化

        # 指纹是否出现
        img_float = img.astype(np.float32)
        res_img = abs(img0_float - img_float)
        cnt1 = np.sum(res_img > 25)  # 背景点数
        if cnt1 < 15555 / 3:
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
        cer[k][0] = find_row(img_a, 0, layer_Row - 1)
        cer[k][1] = find_column(img_a, 0, layer_Column - 1)

        # 计算图像块的背景点数和灰度值之和
        for j in range(0, layer_Column):
            for i in range(0, layer_Row):
                tmp_img = img[i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y].astype(np.float32)
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
                for j in range(cer[k][1], layer_Column):
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

    imwrite(projectPath + 'Experiments/results/algorithm_1/' + str(fileNumber) + '_0' + '.bmp', img0)

    #----------------------------------------------------------------------------
    # 模块二：拼接指纹图像的错位检测
    # 输入：拼接完成的一张指纹图像（img0）
    # 输出：拼接完成的指纹图像的错位情况
    #----------------------------------------------------------------------------


    # 函数similar_Frank 用于计算两个图像块的相似程度
    # ImageBlock_A 与 ImageBlock_B 的尺寸均为 80 * 10
    def similar_Frank(ImageBlock_A, ImageBlock_B, Left):
        # 将图像块进行0-1化预处理
        # Left 为 True 代表，左半部分进行比较，否则右半部分比较
        ImageBlock_A = np.array(ImageBlock_A < 245)
        ImageBlock_B = np.array(ImageBlock_B < 245)
        # 若图像相似，则相异之后，相异为1，相同为0，值0出现的出现次数较多
        numFrank = 0
        if (Left):
            kstart = 0
            kend = 4
        else:
            kstart = 5
            kend = 10
        for i in range(0, 80):
            for j in range(kstart, kend):
                # 统计两个图像块不同的像素点个数，不同的像素点个数越多，拼接就越失败
                if ImageBlock_A[i][j] != ImageBlock_B[i][j]:
                    numFrank = numFrank + 1
        return numFrank

    # 判断水平相邻两个指纹块的缝隙情况
    # 指纹块尺寸：竖直方向80、水平方向10
    # 共计800个像素点
    # 错位率默认是当前块与右边块的图像错位率
    dislocation = np.zeros((layer_Row + 1, layer_Column + 1), dtype=np.float32)
    for i in range(0, layer_Row):
        for j in range(0, layer_Column - 1):
            if result_vis[i][j] == result_vis[i][j + 1]:
                dislocation[i][j] = 0  # 0 表示第i行j列的图像块和第i行j列的图像块，无拼接错位
                continue
            else:
                # 左右相邻两个图像块
                ImageBlock_1 = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y]
                ImageBlock_2 = val_a[result_vis[i][j + 1]][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                # 用以配对的两个图像块
                ImageBlock_3 = val_a[result_vis[i][j + 1]][i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y]
                ImageBlock_4 = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                # 配对：1与3、2与4
                # 拼接情况分数的计算方式：两个图像块相同位置不同像素点的
                score1 = similar_Frank(ImageBlock_1, ImageBlock_3, False)
                score2 = similar_Frank(ImageBlock_2, ImageBlock_4, True)
                # 计算相邻图像块的错位率
                totalScore = (score1 + score2)
                dislocation[i][j] = totalScore / 800 * 100

    # 框出拼接不好的区域
    # 根据错位率进行拼接图像标记
    # 0%～1%之间，认定为表现优秀
    # 1%～10%之间，认定为表现良好
    # 10%～20%之间，认定为表现一般
    # 20%～40%之间，认定为表现糟糕
    # 40%以上，认定为表现极其糟糕
    def TagDislocation(i, j, imgRGB):
        if (dislocation[i][j] < 1):
            return
        elif (dislocation[i][j] < 10):
            left_up = [i * w_x, j * w_y]
            right_down = [(i + 1) * w_x, (j + 2) * w_y]
            # 注意：横坐标差为列，切勿弄反
            # 表现良好：SeaGreen1
            cv2.rectangle(imgRGB, (left_up[1], left_up[0]), (right_down[1], right_down[0]), (84, 255, 159), 3)
        elif (dislocation[i][j] < 20):
            left_up = [i * w_x, j * w_y]
            right_down = [(i + 1) * w_x, (j + 2) * w_y]
            # 注意：横坐标差为列，切勿弄反
            # 表现一般：Sienna1
            cv2.rectangle(imgRGB, (left_up[1], left_up[0]), (right_down[1], right_down[0]), (255, 130, 71), 3)
        elif (dislocation[i][j] < 40):
            left_up = [i * w_x, j * w_y]
            right_down = [(i + 1) * w_x, (j + 2) * w_y]
            # 表现糟糕：PaleVioletRed
            cv2.rectangle(imgRGB, (left_up[1], left_up[0]), (right_down[1], right_down[0]), (219, 112, 147), 3)
        else:
            left_up = [i * w_x, j * w_y]
            right_down = [(i + 1) * w_x, (j + 2) * w_y]
            # 表现极其糟糕：Red
            cv2.rectangle(imgRGB, (left_up[1], left_up[0]), (right_down[1], right_down[0]), (255, 0, 0), 3)

    #----------------------------------------------------------------------------
    # 模块三：拼接指纹图像的质量优化
    # 输入：完整指纹图像各区域的错位情况
    # 输出：进行相邻块优化替换之后的完整指纹图像
    #----------------------------------------------------------------------------

    # 显示拼接完成的图像
    # 注意：具体的框法，需要日后优化
    # 简单优化方法：一张完整图像有512块小图像，假定从左往右滚动
    # 主次目标优化
    for i in range(0, layer_Row):
        for j in range(0, layer_Column - 3):
            if (dislocation[i][j] >= 1):
                for k in range(st+1, ed+1):
                    if useful[k] == 0:  # 未被使用的帧图像，自动忽略
                        continue
                    # 相邻三块：左块、中块、右块
                    # 左右相邻两个图像块，左图像块不变
                    ImageBlock_1 = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y]
                    ImageBlock_2 = val_a[k][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                    # 用以配对的两个图像块
                    ImageBlock_3 = val_a[k][i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y]
                    ImageBlock_4 = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                    # 配对：1与3、2与4
                    # 拼接情况分数的计算方式：两个图像块相同位置不同像素点的
                    score1 = similar_Frank(ImageBlock_1, ImageBlock_3, False)
                    score2 = similar_Frank(ImageBlock_2, ImageBlock_4, True)
                    # 计算替换块，左边的相邻图像块的错位率
                    disRate_Left = (score1 + score2) / 800 * 100

                    # 左右相邻两个图像块，右图像块不变
                    ImageBlock_5 = val_a[k][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                    ImageBlock_6 = val_a[result_vis[i][j + 2]][i * w_x:(i + 1) * w_x, (j + 2) * w_y:((j + 2) + 1) * w_y]
                    # 用以配对的两个图像块
                    ImageBlock_7 = val_a[result_vis[i][j + 2]][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                    ImageBlock_8 = val_a[k][i * w_x:(i + 1) * w_x, (j + 2) * w_y:((j + 2) + 1) * w_y]
                    # 配对：5与7、6与8
                    # 拼接情况分数的计算方式：两个图像块相同位置不同像素点的
                    score3 = similar_Frank(ImageBlock_5, ImageBlock_7, False)
                    score4 = similar_Frank(ImageBlock_6, ImageBlock_8, True)
                    # 计算替换块，左边的相邻图像块的错位率
                    disRate_Right = (score3 + score4) / 800 * 100

                    if (disRate_Left < dislocation[i][j] and disRate_Right < dislocation[i][j + 1]):
                        result_vis[i][j + 1] = k
                        dislocation[i][j] = disRate_Left
                        dislocation[i][j + 1] = disRate_Right

    # 优化后重新赋值
    for j in range(0, layer_Column):
        for i in range(0, layer_Row):
            img0[i * w_x:(i + 1) * w_x,
            j * w_y:(j + 1) * w_y] = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x,
                                     j * w_y:(j + 1) * w_y]

    #----------------------------------------------------------------------------
    # 模块四：提取指纹前景图像，去边缘化
    # 输入：完整指纹图像
    # 输出：指纹前景图像
    #----------------------------------------------------------------------------

    # 将最终错位严重的记录下来（除去边缘外）
    # tempImg = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    tempImg = img0
    disBad = np.zeros((layer_Row + 1, layer_Column + 1), dtype=np.float32)
    # 将边缘框进行删减化
    # 删减依据为框内的灰度值众数为白色
    # 而判断白色的依据为背景帧灰度值的均值
    white = np.mean(img0_float)
    for i in range(0, layer_Row):
        for j in range(0, layer_Column):
            disBad[i][j] = 0
            if (dislocation[i][j] < 1):
                continue
            else:
                ImageBlock_1 = val_a[result_vis[i][j]][i * w_x:(i + 1) * w_x, j * w_y:(j + 1) * w_y]
                ImageBlock_2 = val_a[result_vis[i][j + 1]][i * w_x:(i + 1) * w_x, (j + 1) * w_y:((j + 1) + 1) * w_y]
                mean1 = np.mean(ImageBlock_1)
                mean2 = np.mean(ImageBlock_2)
                # 统计白色点总数
                whitePoint = np.sum(ImageBlock_1 > mean1) + np.sum(ImageBlock_2 > mean2)
                whiteRate = whitePoint / 1600 * 100
                # print(whiteRate)
                if (whiteRate > 60 or mean1 > 200 or mean2 > 200):
                    continue
                else:
                    disBad[i][j] = 1

    imwrite(projectPath + 'Experiments/results/algorithm_2/' + str(fileNumber) + '_0' + '.bmp', tempImg)
    print(str(fileNumber) + "finish！")
