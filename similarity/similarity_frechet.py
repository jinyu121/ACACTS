# -*- coding: utf-8 -*-

import numpy as np

from feature.feature_multi import FeatureMulti
from similarity.similarity_base import SimiliarityBase


class SimilarityFrechet(SimiliarityBase):
    def calculate(self, image1, image2):
        # 获取特征
        multi_feature_extractor = FeatureMulti()
        _, mu_p, sigma_p = multi_feature_extractor.extract(image1)
        _, mu_s, sigma_s = multi_feature_extractor.extract(image2)

        # 实际相似度计算
        return self.__class__.calculate_inner(mu_p, mu_s, sigma_p, sigma_s)

    @classmethod
    def calculate_inner(cls, mu_p, mu_s, sigma_p, sigma_s):
        part_1 = np.power(np.linalg.norm(mu_p - mu_s, 2), 2)
        part_2 = sigma_p + sigma_s - 2 * np.sqrt(sigma_p.dot(sigma_s) + np.finfo(float).eps)
        ans = np.sqrt(part_1 + part_2)
        return np.max(ans)
