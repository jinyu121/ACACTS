# -*- coding: utf-8 -*-
import os

import numpy as np
import skimage.io
from sklearn.externals import joblib

import config
import util as helper
from feature.feature_deep import FeatureDeep
from feature.feature_multi import FeatureMulti
from similarity.similarity_frechet import SimilarityFrechet
from transfer.transfer_all import TransferAll

if '__main__' == __name__:
    # 获取配置
    top_k = config.search['top_k']

    # 读入图片
    image_original = skimage.img_as_float(skimage.io.imread('sample/image/4.jpg'))
    image_original = helper.to_3_dim(image_original)

    # 提取深度特征
    deep_feature_extractor = FeatureDeep()
    feature = deep_feature_extractor.extract(image_original)
    # 提取其他特征
    multi_feature_extractor = FeatureMulti()
    feature_luminance, feature_mu, feature_sigma = multi_feature_extractor.extract(image_original)

    # 如果使用了PCA，那么需要再特殊转换一下
    if config.USE_PCA:
        pca = joblib.load('data/object_pca.pkl')
        feature = pca.transform(feature).ravel()

    # 判断是在哪个语义类里面
    kmeans = joblib.load('data/object_kmeans.pkl')
    # 如果是多类混合
    clusters = np.sum(np.power(kmeans.cluster_centers_ - feature, 2), axis=1).argsort()[:top_k]
    print('Cluster', clusters)

    # 把这些语义类找出来，并得到这个语义类所对应的 K 个“好的”图片
    ranking_all = np.array(joblib.load('data/ranking.pkl'))
    ranking_tops = list()

    # 从每个类里面取出来前K个分数高的好的图片
    for ith_cluster in clusters:
        ranking_tops.append(ranking_all[ith_cluster].argsort()[::-1].ravel()[:top_k])
    ranking_tops = np.unique(np.array(ranking_tops).ravel())

    # 使用其他公式，在做一次ranking
    # 首先把特征加载进来，不用再次计算了
    feature_good = joblib.load('data/ranking_good_feature.pkl')

    # 然后依次计算相似度
    frechet_score = list()
    for ith in ranking_tops:
        frechet_score.append(SimilarityFrechet.calculate_inner(feature_mu,
                                                               feature_good['image_feature_mu'][ith],
                                                               feature_sigma,
                                                               feature_good['image_feature_sigma'][ith])
                             )
    frechet_score = np.array(frechet_score).ravel()
    frechet_score_index_select = np.array(np.where(frechet_score < 7.5)).ravel()
    frechet_score_sort = frechet_score[frechet_score_index_select].argsort().ravel()
    frechet_score_selected = ranking_tops[frechet_score_index_select[frechet_score_sort]][:top_k]

    # 把文件名找出来
    filename_all = np.array(joblib.load('data/data_filename_good.pkl'))
    good_images = filename_all[frechet_score_selected]

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
