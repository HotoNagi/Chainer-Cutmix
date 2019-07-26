# -*- coding: utf-8 -*-
import numpy as np
import cupy as cp
import chainer
import chainercv
from chainercv import transforms
from chainercv.datasets import DirectoryParsingLabelDataset
from chainer.datasets import get_cifar10
import random
import math


# train用
def _train_transform(data):
    # dataには[(img1,label1),(img2,label2)],...,(imgn,labeln)]みたいな感じで
    # 画像とラベルのlistがタプルにまとまって入っている
    # そのdataを一旦imgとdataに分割
    img, lable = data
    # ランダム回転することでデータ数を水増しする
    img = chainercv.transforms.random_rotate(img)
    # ランダムフリップ（反転）することでデータ数を水増しする
    img = chainercv.transforms.random_flip(img, x_random=True, y_random=True)
    return img, lable


# validation用
def _validation_transform(data):
    # dataには[(img1,label1),(img2,label2)],...,(imgn,labeln)]みたいな感じで
    # 画像とラベルのlistがタプルにまとまって入っている
    # そのdataを一旦imgとdataに分割
    img, lable = data

    return img, lable

dataset, _ = get_cifar10()
def _train_cutmix_transform(data):
    # cutmix α for beta
    cutmix_alpha = 0.2
    # class number
    num_class = 10
    # resize coeficient
    h = 32
    w = 32

    img_1, label_1 = data
    img_2, label_2 = random.choice(dataset)


    #####################################################
    # cutmix実装
    #####################################################

    # sample the bounding box coordinates
    while True:
        # the combination ratio λ between two data points is 
        # sampled from the beta distribution Beta(α, α)
        l = np.random.beta(cutmix_alpha, cutmix_alpha)
        rx = random.randint(0, w)
        rw = w * math.sqrt(1-l)
        ry = random.randint(0, h)
        rh = h * math.sqrt(1-l)
        if ((ry + round(rh)) < h)and((rx + round(rw)) < w):
            break

    # denotes a binary mask indicating where to drop out
    # and fill in from two images
    M_1 = np.zeros((h, w))
    M_1[ry:(ry + round(rh)),rx:(rx + round(rw))] = 1

    M = np.ones((h, w))
    M = M - M_1

    # Define the combining operation.
    # img = img_1 * M + img_2 * M_1
    img = (img_1 * M + img_2 * M_1).astype(np.float32)

    # eye関数は単位行列を生成する
    eye = np.eye(num_class)
    label = (eye[label_1] * l + eye[label_2] * (1 - l)).astype(np.float32)

    return img, label