# -*- coding: utf-8 -*-
"""
 @Time    : 19-9-22 下午7:11
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : test_mixup.py
"""
import glob as gb

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Mixup import mixup


def getData():
    batch_x = []
    size = 400
    img_path = gb.glob("./images/*.jpg")
    id = 0

    for ip in img_path:
        img = Image.open(ip)
        img = img.convert('RGB')
        img = img.resize((size, size), Image.ANTIALIAS)
        img = np.array(img)
        img = img[:, :, ::-1]
        batch_x.append(img)
    batch_y = np.arange(0, 8, 1)
    batch_x = np.array(batch_x)
    return batch_x, batch_y


def convert_to_one_hot(y, C):
    return np.around(np.eye(C)[y.reshape(-1)].T,decimals=5)


if __name__ == '__main__':
    alpha = 0.5

    batch_x, batch_y = getData( )
    batch_y = convert_to_one_hot(batch_y, 8)

    batch_x, batch_y = mixup(alpha, batch_x, batch_y)
    batch_y = np.around(batch_y,decimals=5)

    np.savetxt('./result/batch_y_alpha(0.5).csv',batch_y,delimiter=',')

    fig = plt.figure(figsize=(30, 16))
    for i in range(0, 8):
        ax = fig.add_subplot(2, 4, i + 1)
        plt.imshow(batch_x[i].astype(np.uint8))
        plt.axis('off')
    plt.savefig('./result/batch_x_alpha(0.5).jpg')
    plt.show( )






