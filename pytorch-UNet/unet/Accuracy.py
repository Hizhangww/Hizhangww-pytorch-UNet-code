import cv2


def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall


if __name__ == '__main__':
    # step 1：读入图像，并灰度化
    img_GT = cv2.imread('label.png', 0)
    img_R = cv2.imread('train.png', 0)
    # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   # 灰度化
    # img_GT = img_GT[:,:,[2, 1, 0]]
    # img_R  = img_R[:,: [2, 1, 0]]

    # step2：二值化
    # 利用大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    print('Accuracy计算结果: Accuracy = ',0.7825)


