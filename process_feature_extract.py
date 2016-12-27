# -*- coding: utf-8 -*-

import os
from threading import Thread

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
from sklearn.externals import joblib

import config
from feature.feature_multi import FeatureMulti

ans = np.zeros((1, 1))

image_filename = list()
tmp_feature_luminance = list()
tmp_feature_mu = list()
tmp_feature_sigma = list()
job_index = [[], [], [], [], []]

# 特征提取器
multi_feature_extractor = FeatureMulti()


def extract(jobs, base_path):
    global image_filename
    ith = 0
    n_jobs = len(jobs)
    print('Get', n_jobs, 'Jobs')
    for one_index in jobs:
        ith += 1
        one_image = image_filename[one_index]
        try:
            image = skimage.color.rgb2lab(
                skimage.io.imread(os.path.join(base_path, one_image))
            )
            f_luminance, f_mu, f_sigma = multi_feature_extractor.extract(image)
            tmp_feature_luminance[one_index] = f_luminance.copy()
            tmp_feature_mu[one_index] = f_mu.copy()
            tmp_feature_sigma[one_index] = f_sigma.copy()
            print(ith, '/', n_jobs, '\t', one_image)
        except:
            print(ith, '/', n_jobs, '\t', one_image, '\t', 'Error')


def extract_feature(name, base_path):
    global image_filename
    global tmp_feature_luminance
    global tmp_feature_mu
    global tmp_feature_sigma
    global job_index

    # 文件名
    if 'good' == name:
        filename_path = 'data/data_filename_good.pkl'
    else:
        filename_path = 'data/data_filename.pkl'

    if os.path.isfile(filename_path):
        image_filename = joblib.load(filename_path)
    else:
        # 获得文件名
        image_filename = list()
        for parent, dirnames, filenames in os.walk(base_path):
            for f in sorted(filenames):
                if '.jpg' == os.path.splitext(f)[1].lower():
                    image_filename.append(f)
        # 保存文件名
        joblib.dump(image_filename, filename_path)

    # 特征
    n_image = len(image_filename)
    tmp_feature_luminance = [0] * n_image
    tmp_feature_mu = [0] * n_image
    tmp_feature_sigma = [0] * n_image

    # 分配任务
    job_index = [[], [], [], [], []]
    for ith_image in range(n_image):
        job_index[ith_image % 5].append(ith_image)

    # 执行任务
    thread_list = list()
    for ith_thread in range(5):
        thread_list.append(Thread(target=extract, args=(job_index[ith_thread], base_path)))
    for ith_thread in range(5):
        thread_list[ith_thread].start()
    for ith_thread in range(5):
        thread_list[ith_thread].join()

    # 保存特征
    joblib.dump({
        'image_feature_luminance': tmp_feature_luminance,
        'image_feature_mu': tmp_feature_mu,
        'image_feature_sigma': tmp_feature_sigma,
    }, 'data/ranking_' + name + '_feature.pkl')


image_good_dir = config.PATH_IMAGE_GOOD
extract_feature('good', image_good_dir)
image_bad_dir = config.PATH_IMAGE_BAD
extract_feature('bad', image_bad_dir)
