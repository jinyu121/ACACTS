# -*- coding: utf-8 -*-

import os

import numpy as np
import skimage
import skimage.color
import skimage.io


def debug_dump(image, filename='foo.jpg', lab=False, denorm=False):
    im = image.copy()
    if lab:
        if denorm:
            im = lab_denorm(im)
        im = skimage.color.lab2rgb(im)
    skimage.io.imsave(os.path.join('debug', filename), im)


def to_3_dim(image):
    """
    单通道图片变成三通道图片
    """
    if (2 == image.ndim) or (1 == image.shape[2]):
        image = skimage.color.gray2rgb(image)
    return image


def to_3_flat(image):
    """
    将一个图片展开成三维行向量
    """
    return image.reshape(image.shape[0] * image.shape[1], 3)


def lab_norm(image_lab, force_copy=True):
    """
    lab图像正则化
    """
    if force_copy:
        image = image_lab.copy()
    else:
        image = image_lab
    image[:, :, 0] = (image[:, :, 0] / 100)
    image[:, :, 0] = np.minimum(image[:, :, 0], 1)
    image[:, :, 0] = np.maximum(image[:, :, 0], 0)

    image[:, :, 1:3] = ((image[:, :, 1:3] + 127) / 256)
    image[:, :, 1:3] = np.minimum(image[:, :, 1:3], 1)
    image[:, :, 1:3] = np.maximum(image[:, :, 1:3], 0)
    return image


def lab_denorm(image_lab, force_copy=True):
    """
    lab图像反正则化
    """
    if force_copy:
        image = image_lab.copy()
    else:
        image = image_lab
    image[:, :, 0] = (image[:, :, 0] * 100)
    image[:, :, 0] = np.minimum(image[:, :, 0], 100)
    image[:, :, 0] = np.maximum(image[:, :, 0], 0)

    image[:, :, 1:3] = (image[:, :, 1:3] * 256 - 127)
    image[:, :, 1:3] = np.minimum(image[:, :, 1:3], 127)
    image[:, :, 1:3] = np.maximum(image[:, :, 1:3], -127)
    return image


def gaussian_mask(size, sigma):
    """
    制作Gaussian卷积核
    """
    m, n = size
    h, k = m // 2, n // 2
    x, y = np.mgrid[-h:h, -k:k]
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
