# -*- coding: utf-8 -*-

from threading import Thread

import numpy as np
from sklearn.externals import joblib

import config
from similarity.similarity_style import SimilarityStyle

ans = np.zeros((1, 1))

good_filename = list()
good_luminance = list()
good_mu = list()
good_sigma = list()

bad_filename = list()
bad_luminance = list()
bad_mu = list()
bad_sigma = list()


def extract_feature(name):
    if 'good' == name:
        filename_path = 'data/data_filename_good.pkl'
    else:
        filename_path = 'data/data_filename.pkl'

    # 文件名
    image_filename = joblib.load(filename_path)
    # 直接加载特征
    feature_data = joblib.load('data/ranking_' + name + '_feature.pkl')

    return image_filename, \
           feature_data['image_feature_luminance'], \
           feature_data['image_feature_mu'], \
           feature_data['image_feature_sigma']


def calculate(ith_cluster, ith_good, jobs):
    global ans
    global good_luminance
    global good_mu
    global good_sigma
    global bad_luminance
    global bad_mu
    global bad_sigma
    for ith_bad_inner in jobs:
        try:
            score = SimilarityStyle.calculate_inner(
                good_luminance[ith_good], bad_luminance[ith_bad_inner],
                good_mu[ith_good], [ith_bad_inner],
                good_sigma[ith_good], bad_sigma[ith_bad_inner]
            )
            ans[ith_cluster, ith_good] += score
            print('Cluster', ith_cluster, '\t', 'Good', ith_good, '\t', 'Bad', ith_bad_inner, '\t', score)
        except:
            print('Cluster', ith_cluster, '\t', 'Good', ith_good, '\t', 'Bad', ith_bad_inner, '\t', 'Error')


# 设置Good图和Bad图的文件夹
image_good_dir = config.PATH_IMAGE_GOOD
image_bad_dir = config.PATH_IMAGE_BAD
scale = config.style_ranking['scaling']

# 提取特征
good_filename, good_luminance, good_mu, good_sigma = extract_feature('good')
n_good = len(good_filename)

bad_filename, bad_luminance, bad_mu, bad_sigma = extract_feature('bad')
n_bad = len(bad_filename)

# 加载聚类结果
clusters = joblib.load('data/class_index.pkl')
n_clusters = len(clusters)

ans = np.zeros((n_clusters, n_good))

# 对于每一个类
for ith_cluster in range(n_clusters):
    one_cluster = clusters[ith_cluster]
    n_one_cluster = len(one_cluster)
    # 对于每一张好的图片
    for ith_good in range(n_good):
        # 分配任务
        job_index = [[], [], [], [], []]
        for ith_bad in range(n_one_cluster):
            job_index[ith_bad % 5].append(one_cluster[ith_bad])
        # 执行任务
        thread_list = list()
        for ith_thread in range(5):
            thread_list.append(Thread(target=calculate, args=(ith_cluster, ith_good, job_index[ith_thread])))
        for ith_thread in range(5):
            thread_list[ith_thread].start()
            thread_list[ith_thread].join()

# 保存结果
joblib.dump(ans, 'data/ranking.pkl')
