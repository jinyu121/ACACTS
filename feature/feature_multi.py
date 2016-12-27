# -*- coding: utf-8 -*-

import numpy as np

import util as helper
from feature.feature_base import FeatureBase
from feature.feature_historgam_sample import FeatureHistogramSample


class FeatureMulti(FeatureBase):
    def __init__(self):
        self.histogram_sampler = FeatureHistogramSample()

    def extract(self, image):
        # 正则化
        image_lab = helper.lab_norm(image)
        # 求亮度采样
        luminance_sample = self.histogram_sampler.extract(image_lab[:, :, 0])
        # 取出色彩层
        color_layer = helper.to_3_flat(image_lab)[:, 1:3]
        # 计算均值
        mu = np.mean(color_layer, axis=0)
        # 计算方差
        sigma = np.cov(color_layer.transpose())

        return luminance_sample.copy(), mu.copy(), sigma.copy()
