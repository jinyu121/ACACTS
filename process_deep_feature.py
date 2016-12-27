# -*- coding: utf-8 -*-

import os

import numpy as np
import skimage.io
from sklearn.decomposition import PCA
from sklearn.externals import joblib

import config
from feature.feature_deep import FeatureDeep


def process_deep_feature():
    # 构造特征提取器
    extractor = FeatureDeep()

    # 设定路径
    image_dir = config.PATH_IMAGE_BAD

    # 特征数组
    jar_feature = list()
    jar_filename = list()
    for parent, dirnames, filenames in os.walk(image_dir):
        # 对于每一个文件
        total_len = len(filenames)
        ith = 0
        for filename in sorted(filenames):
            ith += 1
            if '.jpg' == os.path.splitext(filename)[1].lower():
                print(ith, '/', total_len, ' ', filename, end="\t")
                try:
                    # 读取图片
                    image = skimage.io.imread(os.path.join(image_dir, filename))
                    # 提取特征
                    f = extractor.extract(image)
                    # 临时保存
                    jar_feature.append(f.copy())
                    jar_filename.append(filename)
                    print('OK')
                except Exception as e:
                    print("Error:", filename)

    # 特征写入磁盘
    jar_feature = np.array(jar_feature)
    joblib.dump(jar_filename, os.path.join('data', 'data_filename.pkl'))
    joblib.dump(jar_feature, os.path.join('data', 'data_feature.pkl'))

    # 顺手做个PCA
    pca = PCA(n_components=512)
    feature_pca = pca.fit_transform(jar_feature)
    joblib.dump(pca, os.path.join('data', 'object_pca.pkl'))
    joblib.dump(feature_pca, os.path.join('data', 'data_feature_pca.pkl'))


if '__main__' == __name__:
    process_deep_feature()
