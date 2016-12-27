# -*- coding: utf-8 -*-

import numpy as np

from feature.feature_base import FeatureBase


class FeatureHistogramSample(FeatureBase):
    def extract(self, image, bins=256, num_of_samples=32):
        [ys, xs] = np.histogram(image, bins=bins, normed=True)
        ys = np.cumsum(ys) / bins
        percents = np.arange(1, 1 + num_of_samples) / num_of_samples
        index = np.searchsorted(ys, percents)
        return xs[index]
