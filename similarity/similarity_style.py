# -*- coding: utf-8 -*-

import numpy as np

import config
from feature.feature_multi import FeatureMulti
from similarity.similarity_base import SimiliarityBase


class SimilarityStyle(SimiliarityBase):
    def calculate(self, image1, image2):
        # 获取特征
        multi_feature_extractor = FeatureMulti()
        luminance_sample_p, mu_p, sigma_p = multi_feature_extractor.extract(image1)
        luminance_sample_s, mu_s, sigma_s = multi_feature_extractor.extract(image2)

        # 实际相似度计算
        return self.__class__.calculate_inner(luminance_sample_p, luminance_sample_s, mu_p, mu_s, sigma_p, sigma_s)

    @classmethod
    def calculate_inner(cls, luminance_sample_p, luminance_sample_s, mu_p, mu_s, sigma_p, sigma_s):
        # 参数
        lambda_l = config.style_ranking['lambda_l']
        lambda_c = config.style_ranking['lambda_c']
        epsilon = config.style_ranking['epsilon']

        # 求亮度之间的欧式距离
        de = np.power(np.linalg.norm(luminance_sample_p - luminance_sample_s, 2), 2)

        # 求色彩之间的距离
        mu = np.matrix(np.abs(mu_p - mu_s) + epsilon).T
        sigma = np.matrix((sigma_p + sigma_s) / 2)

        dh_1 = np.power(np.linalg.norm(sigma_s.dot(sigma_p), 1), 1 / 4) / (np.power(np.linalg.norm(sigma, 1), 1 / 2))
        dh_2 = (-1 / 8) * mu.T * np.linalg.inv(sigma) * mu
        dh = 1 - dh_1 * np.exp(dh_2)

        ans = np.exp(-de / lambda_l) * np.exp(-np.power(dh, 2) / lambda_c)

        # 因为ans是一个 1x1 Matrix，所以必须弄成一个值
        return np.max(ans)
