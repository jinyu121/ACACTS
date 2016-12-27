# -*- coding: utf-8 -*-
import os

import numpy as np
import skimage.io
from sklearn.externals import joblib

import config
import util as helper
from feature.feature_deep import FeatureDeep
from transfer.transfer_all import TransferAll

if '__main__' == __name__:
    # 获取配置
    top_k = config.search['top_k']

    # 读入图片
    image_original = skimage.img_as_float(skimage.io.imread('sample/image/3.jpg'))
    image_original = helper.to_3_dim(image_original)

    # 提取深度特征
    deep_feature_extractor = FeatureDeep()
    feature = deep_feature_extractor.extract(image_original)
    # 如果使用了PCA，那么需要再特殊转换一下
    if config.USE_PCA:
        pca = joblib.load('data/object_pca.pkl')
        feature = pca.transform(feature).ravel()

    # 判断是在哪个语义类里面
    kmeans = joblib.load('data/object_kmeans.pkl')
    # 如果是单类（为了简化问题）
    cluster = kmeans.predict([feature])
    print('Cluster', cluster)

    # 把这个语义类找出来，并得到这个语义类所对应的 K 个“好的”图片
    ranking_all = np.array(joblib.load('data/ranking.pkl'))
    ranking_one = ranking_all[cluster].copy()
    ranking_order = ranking_one.argsort()[::-1].ravel()[:top_k]

    # 把文件名找出来
    filename_all = np.array(joblib.load('data/data_filename_good.pkl'))
    good_images = filename_all[ranking_order]

    # 开始做变换吧！
    transfer = TransferAll()
    ith = 0
    for filename in good_images:
        ith += 1
        # 首先把Good图片读出来
        print("Reference", filename)
        image_sample = skimage.img_as_float(skimage.io.imread(os.path.join(config.PATH_IMAGE_GOOD, filename)))
        # 然后和原图一起做变换
        image_result = transfer.transfer(image_original=image_original, image_sample=image_sample)
        # 结果保存
        # skimage.io.imshow(image_result)
        skimage.io.imsave('result/' + str(ith) + '.jpg', image_result)
