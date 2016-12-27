# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy.optimize
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform

import config
import util as helper
from feature.feature_historgam_sample import FeatureHistogramSample
from .transfer_base import TransferBase


class TransferStyle(TransferBase):
    """
    样式变换
    """

    def __init__(self, image_original, image_sample):
        super(TransferStyle).__init__()
        self.image_original = self.__class__.__transfer_pre(image_original).copy()
        self.image_sample = self.__class__.__transfer_pre(image_sample).copy()
        self.image_result = self.image_original.copy()

    def transfer(self, image_original=None, image_sample=None):
        self.__transfer_color()
        if config.USE_LUMINANCE_CORRECT:
            self.__transfer_luminance()
        self.__transfer_face()

    @classmethod
    def __transfer_pre(cls, image):
        # 参数
        gamma = config.transfer_pre['gamma']
        drop_rate = config.transfer_pre['drop_rate']

        # 做Gamma校正，压缩动态范围
        image = skimage.exposure.adjust_gamma(image, gamma=gamma)

        # 转换到CIELab色域
        image_lab = skimage.color.rgb2lab(image)
        # 做归一化
        # image_lab = helper.lab_norm(image_lab)
        # 取出灰度通道
        image_luminance = image_lab[:, :, 0]

        # 找出亮度最高的5%和最低的5%并扔掉
        image_luminance_flat = image_luminance.ravel().copy()
        image_luminance_sorted = image_luminance_flat.argsort()
        luminance_cut_position = int(image_luminance_flat.size * drop_rate)
        luminance_min = image_luminance_flat[image_luminance_sorted[luminance_cut_position]]
        luminance_max = image_luminance_flat[
            image_luminance_sorted[image_luminance_sorted.size - luminance_cut_position]
        ]
        image_luminance[image_luminance < luminance_min] = luminance_min
        image_luminance[image_luminance > luminance_max] = luminance_max

        # 灰度级拉伸
        image_luminance = (image_luminance - luminance_min) / (luminance_max - luminance_min) * 100
        image_lab[:, :, 0] = image_luminance.copy()

        # 以归一化的Lab返回
        helper.debug_dump(image_lab, lab=True)
        return image_lab

    def __transfer_color(self):
        """
        色彩变换
        """
        x0 = helper.to_3_flat(self.image_original)
        x1 = helper.to_3_flat(self.image_sample)
        a = np.cov(x0.transpose())
        b = np.cov(x1.transpose())

        # 得到变换矩阵
        t = self.__color_helper_mkl(a, b)

        mu_i = np.repeat([x0.mean(axis=0)], x0.shape[0], axis=0)
        mu_s = np.repeat([x1.mean(axis=0)], x0.shape[0], axis=0)

        # 做色彩变换
        xr = (x0 - mu_i).dot(t) + mu_s

        self.image_result = np.array(xr).reshape(self.image_original.shape).copy()

    def __transfer_luminance(self):
        """
        亮度变换
        """
        tau = config.transfer_luminance['tau']
        num_of_samples = config.transfer_luminance['num_of_samples']

        # 采样
        histogram_sampler = FeatureHistogramSample()
        luminance_image = histogram_sampler.extract(self.image_original[:, :, 0], num_of_samples=num_of_samples)
        luminance_sample = histogram_sampler.extract(self.image_sample[:, :, 0], num_of_samples=num_of_samples)

        luminance_calculated = luminance_image + (luminance_sample - luminance_image) * (
            tau / np.minimum(tau, np.linalg.norm(luminance_sample - luminance_image, np.inf)))

        target_function = lambda para: self.__class__.__luminance_helper_cost(para, luminance_image,
                                                                              luminance_calculated)
        result = scipy.optimize.minimize(target_function, np.random.random_sample([2]), options={'disp': True})

        self.image_result[:, :, 0] = self.__class__.__luminance_helper_transfer_function(self.image_result[:, :, 0],
                                                                                         result.x)

    def __transfer_face(self):
        # 获得参数
        luminance_th_r = config.transfer_face['luminance_th_r']
        gamma_th = config.transfer_face['gamma_th']
        alpha_r = config.transfer_face['alpha_r']
        alpha_c = config.transfer_face['alpha_c']

        # 获取灰度通道
        image = self.image_result[:, :, 0]

        # 生成人脸检测器
        face_cascade = cv2.CascadeClassifier(config.PATH_OPENCV)

        # 人脸检测
        faces = face_cascade.detectMultiScale(skimage.img_as_ubyte(image / np.max(image)), 1.3, 5)

        # 对于每一个人脸进行处理
        for (x, y, w, h) in faces:
            # 框框的边界
            box_u = y
            box_d = y + h
            box_l = x
            box_r = x + w
            box_center = np.array([x + w / 2, y + h / 2])
            radius = max(w, h) / 2

            # 获取全通道图和亮度图
            im_small = self.image_result[box_u:box_d, box_l:box_r, :]
            im_small_luminance = im_small[:, :, 0]

            # 计算出一些统计信息
            luminance_max = np.max(im_small_luminance)
            luminance_min = np.min(im_small_luminance)
            luminance_mean = np.median(im_small_luminance)
            color_mean = np.median(np.median(im_small, axis=0), axis=0)
            luminance_th = (luminance_max - luminance_min) * luminance_th_r + luminance_min

            # 如果亮度不足，那么进行亮度补偿
            if luminance_mean < luminance_th:
                # 一个新的Gamma，用于Gamma校正
                gamma = max(gamma_th, luminance_mean / luminance_th)
                # 计算权重
                [xx, yy] = np.mgrid[box_l:box_r, box_u:box_d]
                w1 = np.exp(-alpha_r * (np.power(np.abs(xx - box_center[0]) + np.abs(yy - box_center[1]), 2) /
                                        np.power(radius, 2)))
                w2 = np.exp(-alpha_c * np.sum(np.power(im_small - color_mean, 0), axis=2))
                weight = -w1 * w2

                # 为了解决亮度校正过于突兀的问题，生成一个 Gaussian Mask
                mask_size = max(w, h)
                mask = helper.gaussian_mask([mask_size, mask_size], mask_size / 4)
                mask = skimage.transform.resize(mask, (h, w))

                im_small_luminance = (1 - weight) * im_small_luminance + weight * (np.power(im_small_luminance, gamma))
                self.image_result[box_u: box_d, box_l: box_r, 0] = (
                    im_small[:, :, 0] * (1 - mask) + im_small_luminance * mask
                ).copy()

    def get_result(self):
        """
        获取变换后的结果
        """
        gamma = 1 / config.transfer_pre['gamma']
        image = skimage.color.lab2rgb(self.image_result)
        return skimage.exposure.adjust_gamma(image, gamma=gamma)

    def __color_helper_mkl(self, sigma_image, sigma_sample):
        """
        生成色彩变换矩阵
        """
        lambda_r = config.transfer_color['lambda_r']
        sigma_image = np.maximum(sigma_image, np.eye(sigma_image.shape[0]) * lambda_r)

        [val_i, vec_i] = np.linalg.eig(sigma_image)
        val_i[val_i < 0] = 0
        da = np.diag(np.sqrt(val_i + np.finfo(float).eps))
        c = np.matrix(da) * (np.matrix(vec_i).T) * np.matrix(sigma_sample) * np.matrix(vec_i) * np.matrix(da)
        [val_c, vec_c] = np.linalg.eig(c)
        val_c[val_c < 0] = 0
        dc = np.diag(np.sqrt(val_c + np.finfo(float).eps))
        da_inv = np.diag(1 / (np.diag(da)))
        return (np.matrix(vec_i) * np.matrix(da_inv)) * \
               (np.matrix(vec_c) * np.matrix(dc) * (np.matrix(vec_c).T)) * \
               (np.matrix(da_inv) * (np.matrix(vec_i).T))

    @classmethod
    def __luminance_helper_cost(cls, param, luminance_input, luminance_calculated):
        """
        亮度转换代价函数
        """
        return np.power(
            np.linalg.norm(
                cls.__luminance_helper_transfer_function(
                    luminance_image=luminance_input, param=param
                ) - luminance_calculated,
                2),
            2)

    @classmethod
    def __luminance_helper_transfer_function(cls, luminance_image, param):
        """
        亮度转换函数
        """
        tmp = np.arctan(param[0] / param[1])
        return (tmp + np.arctan((luminance_image - param[0]) / param[1])) / (tmp + np.arctan((1 - param[0]) / param[1]))
