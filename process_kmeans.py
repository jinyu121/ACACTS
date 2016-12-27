# -*- coding: utf-8 -*-

import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.externals import joblib

import config


def process_kmeans():
    # 配置
    n_clusters = config.kmeans['n_clusters']

    # 读数据
    print('Load Data...')
    if config.USE_PCA:
        feature = joblib.load(os.path.join('data', 'data_feature_pca.pkl'))
    else:
        feature = joblib.load(os.path.join('data', 'data_feature.pkl'))
    print('OK')

    # 做KMeans聚类
    print('KMeans Cluster...')

    if feature.shape[0] <= 10000:
        # 最原始的KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, verbose=1).fit(feature)
    else:
        # MiniBatchKMeans ，速度会快一点
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, init_size=3 * n_clusters, verbose=1).fit(feature)
    print('OK')

    # 保存结果
    print('Save Result...')
    joblib.dump(kmeans, 'data/object_kmeans.pkl')
    joblib.dump(kmeans.labels_, 'data/data_kmeans_label.pkl')
    print('OK')

    # 重新做整理，整理成类似于“key-value”的形式
    # 其实是 index-[filename_1, filename_2, ...] 的形式
    print('Re-Arrange Result...')
    data_class = kmeans.labels_.copy()
    data_filename = np.array(joblib.load('data/data_filename.pkl'))
    jar_index = list()
    jar_filename = list()

    for ith in range(np.max(data_class) + 1):
        index = np.where(data_class == ith)[0]
        index.sort()
        print('Class', ith, '\t', len(index))
        print(data_filename[index])

        jar_index.append(index.copy())
        jar_filename.append(data_filename[index].copy())

    joblib.dump(jar_index, 'data/class_index.pkl')
    joblib.dump(jar_filename, 'data/class_filename.pkl')
    print('OK')


if '__main__' == __name__:
    process_kmeans()
